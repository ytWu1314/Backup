# Generated by Django 2.1.2 on 2018-10-13 12:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('search_engine', '0003_auto_20181013_1637'),
    ]

    operations = [
        migrations.AlterField(
            model_name='csdnblog',
            name='url',
            field=models.CharField(max_length=200),
        ),
    ]
