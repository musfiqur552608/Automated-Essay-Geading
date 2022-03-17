from django.db import models

# Create your models here.
class EssayGrade(models.Model):
    mytext = models.CharField(max_length=1000000)
    score = models.FloatField()
    out = models.IntegerField()