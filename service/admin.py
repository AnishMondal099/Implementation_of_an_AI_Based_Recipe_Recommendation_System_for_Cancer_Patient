from django.contrib import admin
from service.models import Service
class Serviceadmin(admin.ModelAdmin):
    list_display=('name','email')
admin.site.register(Service,Serviceadmin)
# Register your models here.
