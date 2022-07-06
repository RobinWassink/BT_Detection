#!/bin/bash

#start monitoring with ./start_monitor.sh

read -p 'Which file has been modified? (rtlsdr/FFT): ' file
read -p 'How long do you want to monitor (in seconds)?: ' duration
read -p "What's the affected bandwith (in kHz)?: " bandwidth

current=`date "+%Y-%m-%d_%H-%M-%S"`;
echo "Start monitoring script at $current" 
folder="/data/$current"_"$file"_"$bandwith"_"$duration/raw"
mkdir $folder

for mode in normal repeat mimic confusion noise spoof freeze delay stop
do
    mount -o remount rw /
    if [ $mode == "stop" ]
		then
			#STOP MONITORING
			echo "Stopping monitoring script..."
			service electrosense-sensor-mqtt stop
            echo "Done."
    else
        #SET MALICIOUS EXECUTABLE AND START MONITORING
        echo "Restarting es_sensor executable with new behavior for $mode"
        service electrosense-sensor-mqtt stop
        cp $mode /usr/bin/es_sensor
        service electrosense-sensor-mqtt start
        sleep 10
        echo "Starting monitoring script"
        ./monitor.sh $mode $current $duration $folder&
        # wait until get trace finished
        wait
        echo "Done."
        echo "================================================================="
        sleep 10
    fi
done
echo "finished!!!"