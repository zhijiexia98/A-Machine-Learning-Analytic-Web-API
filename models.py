from django.db import models

# Create your models here.
class FileModel(models.Model):
    file_name = models.CharField(max_length=50)
    file_content = models.FileField(upload_to="upload")