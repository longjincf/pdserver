# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
from django.db import models
from django.utils import timezone

class AssessmentResult(models.Model):
    ASSESSMENT_TYPES = [
        ('voice', '语音评估'),
        ('face', '面部评估'), 
        ('gait', '步态评估'),
    ]
    
    assessment_type = models.CharField(max_length=20, choices=ASSESSMENT_TYPES)
    features = models.JSONField(verbose_name='特征数据')
    prediction = models.JSONField(verbose_name='预测结果')
    severity_bucket = models.IntegerField(verbose_name='严重程度分级')
    total_updrs = models.FloatField(null=True, blank=True, verbose_name='UPDRS总分')
    created_at = models.DateTimeField(default=timezone.now, verbose_name='创建时间')
    
    class Meta:
        db_table = 'assessment_results'
        verbose_name = '评估结果'
        verbose_name_plural = '评估结果'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_assessment_type_display()} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

