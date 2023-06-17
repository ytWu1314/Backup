# Generated by Django 2.1.2 on 2018-11-02 04:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('search_engine', '0004_auto_20181013_2020'),
    ]

    operations = [
        migrations.CreateModel(
            name='Query',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('query', models.CharField(max_length=100)),
                ('date', models.DateTimeField(verbose_name='date searched')),
            ],
        ),
    ]
