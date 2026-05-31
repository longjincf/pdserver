# Copyright (C) 2025 步健慧测团队 (华南师范大学)
# SPDX-License-Identifier: GPL-3.0-or-later
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    path('face/', TemplateView.as_view(template_name='face.html'), name='face'),
    path('gait/', TemplateView.as_view(template_name='gait.html'), name='gait'),
    path('voice/', TemplateView.as_view(template_name='voice.html'), name='voice'),
]
