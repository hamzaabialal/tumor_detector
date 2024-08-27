from django.db import models

class TumorPrediction(models.Model):
    image_path = models.ImageField(upload_to='images/', blank=False, null=False)
    result = models.FloatField()
    diagnosis = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
