#!/usr/bin/env python3

import rospy
import os
from duckietown_msgs.msg import WheelsCmdStamped # Import the message

THROTTLE = .5


#for the wheel command
class Driver():#CHANGE CLASSNAME to the name of your class
    def __init__(self):
        self.veh_name = os.environ['VEHICLE_NAME']
        # Stages: 0 = move, 1 = stop, 2 = turn
        self.stage = 0


# USING PARAMETER TO GET THE NAME OF THE VEHICLE
# THIS WILL BE USEFUL TO SPECIFY THE NAME OF THE TOPIC
# INITIALIZE YOUR VARIABLES HERE (SUBSCRIBERS OR PUBLISHERS)
    def drive(self): # CHANGE TO THE NAME OF YOUR FUNCTION
        print("Driving the MM " + self.veh_name + " around the block")
        match self.stage:
            cmd = WheelsCmdStamped()
            case 0:
                print("Moving forward")
                cmd._vel_left = THROTTLE *1
                cmd._vel_right = THROTTLE *1 
                rospy.sleep(1.) # Calculate speed later
                cmd._vel_left = 0
                cmd._vel_right = 0 
                
                stage = 1
            case 1:
                print("Waiting")
                rospy.sleep(5.)
                
                stage = 2
            case 2:
                print("Turning")
                cmd._vel_left = THROTTLE *1
                cmd._vel_right = -THROTTLE *1 
                rospy.sleep(1.) # Calculate speed later
                cmd._vel_left = 0
                cmd._vel_right = 0
                
                stage = 0
        wheels_topic = f"/{self.veh_name}/wheels_driver_node/wheels_cmd"
        rospy.Publisher(wheels_topic, cmd, queue_size = 10)

# Just for testing
#WRITE THE CODE TO MAKE THE MM GO AROUND THE BLOCK

if __name__ == "__main__": ## The main function which will be called
# when your python script is executed.
# Initialize the node
    try:
        rospy.init_node('driving')
        drive = Driver() # Create obj of the Driver class
        rospy.sleep(2) # Delay to wait enough time for the code to
#run
# Keep the line above - you might be able to reduce the
# delay a bit,
        while not rospy.is_shutdown(): # Run ros forever - you can change
# this as well instead of running forever
            drive.drive() # calling your node function
    except rospy.ROSInterruptException:
        pass