#!/usr/bin/env python3

import rospy
import os
from duckietown_msgs.msg import WheelsCmdStamped # Import the message

THROTTLE = .5


#for the wheel command
class Driver():#CHANGE CLASSNAME to the name of your class
    def __init__(self):
        self.veh_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{self.veh_name}/wheels_driver_node/wheels_cmd"
        self.wheels_cmd=rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size =10)
        self.timeStart = rospy.Time.now()
        
        # Stages: 0 = move, 1 = stop, 2 = turn
        self.stage = 0


# USING PARAMETER TO GET THE NAME OF THE VEHICLE
# THIS WILL BE USEFUL TO SPECIFY THE NAME OF THE TOPIC
# INITIALIZE YOUR VARIABLES HERE (SUBSCRIBERS OR PUBLISHERS)
    def drive(self): # CHANGE TO THE NAME OF YOUR FUNCTION
        print("Driving the MM " + self.veh_name + " around the block")
        cmd = WheelsCmdStamped()
        if self.stage == 0:
            self.timeStart = rospy.Time.now()
            print("Moving forward")
            while (rospy.Time.now() -self.timeStart).to_sec() < 3:
                cmd.vel_left = THROTTLE *1
                cmd.vel_right = THROTTLE *1 
                self.wheels_cmd.publish(cmd)
            self.stage = 1
        elif self.stage == 1:
            print("Waiting")
            # while (rospy.Time.now() -self.timeStart).to_sec()
            cmd.vel_left = 0
            cmd.vel_right = 0
            self.wheels_cmd.publish(cmd) 
            rospy.sleep(5.)

            self.stage = 2
        elif self.stage == 2:
            print("Turning")
            self.timeStart = rospy.Time.now()
            while (rospy.Time.now() -self.timeStart).to_sec() < .3:
                cmd.vel_left = THROTTLE *0.85
                cmd.vel_right = -THROTTLE *1 
                self.wheels_cmd.publish(cmd)                
            self.stage = 3
        elif self.stage == 3:
            cmd.vel_left = 0
            cmd.vel_right = 0
            self.wheels_cmd.publish(cmd) 
            rospy.sleep(2.)
            self.stage = 0
        print(cmd)



# Just for testing
#WRITE THE CODE TO MAKE THE MM GO AROUND THE BLOCK

if __name__ == "__main__": ## The main function which will be called
# when your python script is executed.
# Initialize the node
    try:
        rospy.init_node('driving')
        mm_driver = Driver() # Create obj of the Driver class
        rospy.sleep(2) # Delay to wait enough time for the code to
#run
# Keep the line above - you might be able to reduce the
# delay a bit,
        i = 0
        while (not rospy.is_shutdown()) and i<16: # Run ros forever - you can change
# this as well instead of running forever
            mm_driver.drive() # calling your node function
            i += 1
    except rospy.ROSInterruptException:
        pass