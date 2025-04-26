from django.db import models
class Service(models.Model):
    name=models.CharField(max_length=50)
    email=models.EmailField(max_length=200,unique=True)
    password=models.CharField(max_length=50)

# Create your models here.
