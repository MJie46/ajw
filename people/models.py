from django.db import models


class Person(models.Model):
    name = models.CharField(max_length=30)
    age = models.IntegerField()

class TrainResult(models.Model):
    step = models.IntegerField()
    resultId = models.IntegerField()
    image = models.BinaryField()
    width = models.IntegerField()
    height = models.IntegerField()