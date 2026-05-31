# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
from django.urls import path
from .views import (
    health, model_info, predict, face_predict, gait_predict, 
    save_data, get_all_data, get_data_by_id, get_data_by_type, get_latest_data
)

urlpatterns = [
    path('health', health),
    path('model_info', model_info),
    path('predict', predict),
    path('face_predict', face_predict),
    path('gait_predict', gait_predict),
    # 数据保存和获取路由
    path('save_data', save_data, name='save_data'),
    path('get_all_data', get_all_data, name='get_all_data'),
    path('get_data/<int:result_id>', get_data_by_id, name='get_data_by_id'),
    path('get_data_by_type/<str:assessment_type>', get_data_by_type, name='get_data_by_type'),
    path('get_latest_data', get_latest_data, name='get_latest_data'),
]

