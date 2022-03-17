from rest_framework import serializers
from .models import EssayGrade

class EssayGradeSerializer(serializers.Serializer):
    mytext = serializers.CharField(max_length=1000000)
    score = serializers.FloatField()
    out = serializers.IntegerField()
    
    # created this function for create, read, delete
    def create(self, validated_data):
        return EssayGrade.objects.create(**validated_data)
    
    # created this function for update
    def update(self, instance, validated_data):
        instance.mytext = validated_data.get('mytext', instance.mytext)
        instance.score = validated_data.get('score', instance.score)
        instance.out = validated_data.get('out', instance.out)
        instance.save()
        return instance