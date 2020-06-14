from crontab import CronTab

my_cron = CronTab(user='amroghoneim')

job = my_cron.new(command = 'python3 /home/amroghoneim/Desktop/job_recommender.py')
job.minute.every(1)
my_cron.write()

for job in my_cron:
    print (job)



