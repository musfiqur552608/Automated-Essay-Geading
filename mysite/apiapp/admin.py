from django.contrib import admin
from .models import EssayGrade

# Register your models here.
@admin.register(EssayGrade)
class EssayGradeAdmin(admin.ModelAdmin):
    list_display = ['id', 'mytext', 'score', 'out']