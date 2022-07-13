
from .models import FileModel
from rest_framework import serializers

class FileSerializer(serializers.ModelSerializer):
    class Meta:
            model = FileModel
            fields = '__all__'