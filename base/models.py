from django.db import models


class Images(models.Model):
    path = models.CharField(max_length=200)
    result = models.CharField(max_length=50)

    def __str__(self):
        return self.path
