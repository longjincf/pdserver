# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
import os, tempfile, json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from joblib import load
from .feature_extractor import extract_features as extract_voice_features
from .feature_extractor_face import analyze_face
from .modeling import load_voice_model, predict_voice, updrs_to_bucket
from .gait_modeling import load_model as load_gait_model, predict_gait as gait_predict_unified, updrs_to_bucket as gait_updrs_to_bucket
from .feature_extractor_gait import analyze_gait_from_video
from .models import AssessmentResult

def health(request):
    ok = os.path.exists(settings.MODEL_PATH)
    return JsonResponse({"status": "ok", "model_ready": ok})

def model_info(request):
    info = {}
    voice_model = load_voice_model()
    if voice_model is not None:
        info['voice_model'] = voice_model.get('meta', {})
        info['voice_model_type'] = voice_model['type']
    gait_model = load_gait_model()
    info['gait_model_ready'] = gait_model is not None
    if gait_model is not None:
        info['gait_model_type'] = gait_model['type']
    from .face_modeling import load_face_lstm_model
    face_model = load_face_lstm_model()
    info['face_model_ready'] = face_model is not None
    if face_model is not None:
        info['face_model_type'] = face_model['type']
    return JsonResponse(info)

@csrf_exempt
def predict(request):
    # 兼容之前的语音预测接口（POST audio）
    if request.method != 'POST':
        return JsonResponse({"error": "POST a .wav/.mp3 file as 'audio'."}, status=405)
    if 'audio' not in request.FILES:
        return JsonResponse({"error": "Missing file field 'audio'."}, status=400)
    f = request.FILES['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        for chunk in f.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    # 提取聚合特征（兼容旧接口）和帧级时序（供 LSTM 使用）
    feats = None
    ts_result = None
    try:
        feats = extract_voice_features(tmp_path, save_plots=False)
    except Exception:
        pass
    try:
        from .feature_extractor import extract_features_timeseries
        ts_result = extract_features_timeseries(tmp_path)
    except Exception:
        pass
    finally:
        try: os.remove(tmp_path)
        except: pass

    if feats is None and ts_result is None:
        return JsonResponse({"error": "语音特征提取失败。"}, status=500)

    # 尝试加载模型（LSTM 优先，RF fallback）
    voice_model = load_voice_model()
    model_type = "heuristic"

    if voice_model is not None:
        # LSTM 优先用时序数据，RF 用聚合特征
        if voice_model["type"] == "lstm" and ts_result is not None:
            updrs_pred = predict_voice(voice_model, ts_result["sequence"])
        elif feats is not None:
            updrs_pred = predict_voice(voice_model, {
                'jitter_local': feats['jitter_local'],
                'shimmer_local': feats['shimmer_local'],
                'hnr_db': feats['hnr_db'],
            })
        else:
            updrs_pred = None
        if updrs_pred is not None:
            bucket = updrs_to_bucket(updrs_pred)
            model_type = voice_model["type"]
        else:
            updrs_pred = None
    else:
        updrs_pred = None

    # 无模型时用启发式映射
    if updrs_pred is None and feats is not None:
        j, s, h = feats['jitter_local'], feats['shimmer_local'], feats['hnr_db']
        score = (j*2 + s*1.5 + max(0, 20-h))
        if score <= 10: bucket = 0
        elif score <= 20: bucket = 1
        elif score <= 30: bucket = 2
        elif score <= 40: bucket = 3
        else: bucket = 4

    response_data = {
        "features": {"jitter_local_percent": round(feats['jitter_local'],4), "shimmer_local_percent": round(feats['shimmer_local'],4), "hnr_db": round(feats['hnr_db'],3)} if feats else {},
        "prediction": {"total_UPDRS": None if updrs_pred is None else round(updrs_pred,2), "severity_bucket": int(bucket)},
        "model_type": model_type,
    }

    # 保存数据到数据库
    try:
        AssessmentResult.objects.create(
            assessment_type='voice',
            features=response_data['features'],
            prediction=response_data['prediction'],
            severity_bucket=bucket,
            total_updrs=updrs_pred
        )
    except Exception as e:
        print(f"保存语音评估数据失败: {e}")

    return JsonResponse(response_data)

@csrf_exempt
def face_predict(request):
    # 接收视频文件 field 名称 'video'
    if request.method != 'POST':
        return JsonResponse({"error": "POST a video file as 'video'."}, status=405)
    if 'video' not in request.FILES:
        return JsonResponse({"error": "Missing file field 'video'."}, status=400)
    f = request.FILES['video']
    suffix = os.path.splitext(f.name)[1].lower()
    if suffix not in ['.mp4','.avi','.mov','.mkv']:
        return JsonResponse({"error": "Unsupported video format. Use mp4/avi/mov/mkv."}, status=400)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in f.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    # 规则打分（始终计算，用于 fallback 和特征返回）
    result = None
    ts_result = None
    try:
        result = analyze_face(tmp_path)
    except Exception:
        pass
    try:
        from .feature_extractor_face import analyze_face_timeseries
        ts_result = analyze_face_timeseries(tmp_path)
    except Exception:
        pass
    finally:
        try: os.remove(tmp_path)
        except: pass

    if result is None and ts_result is None:
        return JsonResponse({"error": "face analysis failed: 无法从视频中检测到人脸。"}, status=500)

    # LSTM 模型优先
    from .face_modeling import load_face_lstm_model, predict_face_grade
    face_model = load_face_lstm_model()
    model_type = "rule"

    if face_model is not None and ts_result is not None:
        try:
            grade, description = predict_face_grade(face_model, ts_result["sequence"])
            # 用 LSTM 结果覆盖规则打分的分级，但保留规则提取的特征
            if result is not None:
                result["grade"] = grade
                result["description"] = description
            else:
                result = {"grade": grade, "description": description,
                          "blink_rate_per_min": 0, "emotion_entropy": 0, "lips_apart_ratio": 0}
            model_type = "lstm"
        except Exception:
            pass  # fallback 到规则打分

    if result is None:
        return JsonResponse({"error": "face analysis failed."}, status=500)

    # 保存数据到数据库
    try:
        AssessmentResult.objects.create(
            assessment_type='face',
            features={
                "blink_rate_per_min": result["blink_rate_per_min"],
                "emotion_entropy": result["emotion_entropy"],
                "lips_apart_ratio": result["lips_apart_ratio"]
            },
            prediction={
                "grade": result["grade"],
                "description": result["description"]
            },
            severity_bucket=result["grade"]
        )
    except Exception as e:
        print(f"保存面部评估数据失败: {e}")

    result["model_type"] = model_type
    return JsonResponse(result)

@csrf_exempt
def gait_predict(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Use POST. Either upload 'video' or send empty POST to use server-side gait model."}, status=405)
    # 上传视频走视频提取路径
    if 'video' in request.FILES:
        f = request.FILES['video']
        suffix = os.path.splitext(f.name)[1].lower()
        if suffix not in ['.mp4','.avi','.mov','.mkv']:
            return JsonResponse({"error": "Unsupported video format. Use mp4/avi/mov/mkv."}, status=400)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in f.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        # 提取聚合特征（兼容旧接口）和原始时序（供 LSTM 使用）
        feats = None
        ts_result = None
        try:
            feats = analyze_gait_from_video(tmp_path)
        except Exception:
            pass
        try:
            from .feature_extractor_gait import extract_gait_timeseries
            ts_result = extract_gait_timeseries(tmp_path)
        except Exception:
            pass
        finally:
            try: os.remove(tmp_path)
            except: pass

        if feats is None and ts_result is None:
            return JsonResponse({"error": "步态特征提取失败，请检查视频质量。"}, status=500)

        gait_model = load_gait_model()
        if gait_model is None:
            return JsonResponse({"features": feats, "message": "Gait model not trained. Set GAIT_DATA_PATH and run `python manage.py train_gait` to train."})

        # LSTM 模型优先用时序数据，RF 用聚合特征
        if gait_model["type"] == "lstm" and ts_result is not None:
            updrs_pred = gait_predict_unified(gait_model, ts_result["sequence"])
        elif feats is not None:
            updrs_pred = gait_predict_unified(gait_model, feats)
        else:
            return JsonResponse({"error": "模型与特征不匹配。"}, status=500)

        bucket = gait_updrs_to_bucket(updrs_pred)
        response_data = {
            "features": feats or {"duration": ts_result["duration"], "n_frames": ts_result["n_frames"], "step_count": ts_result["step_count"]},
            "prediction": {"total_UPDRS": round(updrs_pred, 2), "severity_bucket": int(bucket)},
            "model_type": gait_model["type"],
        }

        try:
            AssessmentResult.objects.create(
                assessment_type='gait',
                features=response_data['features'],
                prediction=response_data['prediction'],
                severity_bucket=bucket,
                total_updrs=updrs_pred
            )
        except Exception as e:
            print(f"保存步态评估数据失败: {e}")

        return JsonResponse(response_data)

    # 空POST -> 返回模型信息或错误
    gait_model = load_gait_model()
    if gait_model is None:
        return JsonResponse({"error": "Gait model not trained. Set GAIT_DATA_PATH env to a CSV and run manage.py train_gait"}, status=400)
    return JsonResponse({"message": "Gait model ready. POST a 'video' file to predict.", "model_type": gait_model["type"], "meta": gait_model.get('meta', {})})

# 新增API：保存数据（用于前端手动保存）
@csrf_exempt
def save_data(request):
    if request.method != 'POST':
        return JsonResponse({"error": "只支持POST请求"}, status=405)
    
    try:
        data = json.loads(request.body)
        assessment_type = data.get('assessment_type')
        features = data.get('features', {})
        prediction = data.get('prediction', {})
        severity_bucket = data.get('severity_bucket', 0)
        total_updrs = data.get('total_updrs')
        
        if not assessment_type:
            return JsonResponse({"error": "缺少评估类型(assessment_type)"}, status=400)
        
        result = AssessmentResult.objects.create(
            assessment_type=assessment_type,
            features=features,
            prediction=prediction,
            severity_bucket=severity_bucket,
            total_updrs=total_updrs
        )
        
        return JsonResponse({
            "success": True,
            "message": "数据保存成功",
            "id": result.id,
            "created_at": result.created_at.isoformat()
        })
        
    except Exception as e:
        return JsonResponse({"error": f"保存数据失败: {str(e)}"}, status=500)

# 新增API：获取所有数据（用于微信小程序）
def get_all_data(request):
    try:
        results = AssessmentResult.objects.all().order_by('-created_at')
        data = []
        for result in results:
            data.append({
                "id": result.id,
                "assessment_type": result.assessment_type,
                "features": result.features,
                "prediction": result.prediction,
                "severity_bucket": result.severity_bucket,
                "total_updrs": result.total_updrs,
                "created_at": result.created_at.isoformat()
            })
        
        return JsonResponse({
            "success": True,
            "data": data,
            "count": len(data)
        })
        
    except Exception as e:
        return JsonResponse({"error": f"获取数据失败: {str(e)}"}, status=500)

# 新增API：根据ID获取单条数据
def get_data_by_id(request, result_id):
    try:
        result = AssessmentResult.objects.get(id=result_id)
        return JsonResponse({
            "success": True,
            "data": {
                "id": result.id,
                "assessment_type": result.assessment_type,
                "features": result.features,
                "prediction": result.prediction,
                "severity_bucket": result.severity_bucket,
                "total_updrs": result.total_updrs,
                "created_at": result.created_at.isoformat()
            }
        })
    except AssessmentResult.DoesNotExist:
        return JsonResponse({"error": "数据不存在"}, status=404)
    except Exception as e:
        return JsonResponse({"error": f"获取数据失败: {str(e)}"}, status=500)

# 新增API：根据评估类型获取数据
def get_data_by_type(request, assessment_type):
    try:
        results = AssessmentResult.objects.filter(
            assessment_type=assessment_type
        ).order_by('-created_at')
        
        data = []
        for result in results:
            data.append({
                "id": result.id,
                "assessment_type": result.assessment_type,
                "features": result.features,
                "prediction": result.prediction,
                "severity_bucket": result.severity_bucket,
                "total_updrs": result.total_updrs,
                "created_at": result.created_at.isoformat()
            })
        
        return JsonResponse({
            "success": True,
            "data": data,
            "count": len(data)
        })
        
    except Exception as e:
        return JsonResponse({"error": f"获取数据失败: {str(e)}"}, status=500)

# 新增API：获取最新数据
def get_latest_data(request):
    try:
        # 获取每种评估类型的最新一条数据
        latest_data = {}
        for assessment_type in ['voice', 'face', 'gait']:
            try:
                latest = AssessmentResult.objects.filter(
                    assessment_type=assessment_type
                ).latest('created_at')
                latest_data[assessment_type] = {
                    "id": latest.id,
                    "features": latest.features,
                    "prediction": latest.prediction,
                    "severity_bucket": latest.severity_bucket,
                    "total_updrs": latest.total_updrs,
                    "created_at": latest.created_at.isoformat()
                }
            except AssessmentResult.DoesNotExist:
                latest_data[assessment_type] = None
        
        return JsonResponse({
            "success": True,
            "data": latest_data
        })
        
    except Exception as e:
        return JsonResponse({"error": f"获取最新数据失败: {str(e)}"}, status=500)


