#! /usr/bin/env python
import rospy
from std_msgs.msg import String
import base64
import socket
import errno
import time
import threading
import select
import Queue
import datetime
import serial

# Ros Messages
from sensor_msgs.msg import NavSatFix

# Global variables
rtcm_queue = Queue.Queue()
gga_queue = Queue.Queue()


class NtripSocketThread (threading.Thread):
    def __init__(self, caster_ip, caster_port, mountpoint, username, password):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.no_rtcm_data_count = 0
        self.sent_gga = False
        self.ntrip_tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected_to_caster = False
        self.username = username
        self.password = password
        self.mountpoint = mountpoint
        self.caster_ip = caster_ip
        self.caster_port = caster_port

    def connect_to_ntrip_caster(self):
        print('Connecting to NTRIP caster at %s:%d' % (self.caster_ip, self.caster_port))

        try:
            self.ntrip_tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ntrip_tcp_sock.settimeout(5.0)
            self.ntrip_tcp_sock.connect((self.caster_ip, self.caster_port))
            self.ntrip_tcp_sock.settimeout(None)
            print('Successfully opened socket')
        except Exception as ex:
            print('Error connecting socket: %s' % ex)
            self.ntrip_tcp_sock.settimeout(None)
            return False

        encoded_credentials = base64.b64encode(self.username + ':' + self.password)
        server_request = 'GET /%s HTTP/1.0\r\nUser-Agent: NTRIP ABC/1.2.3\r\nAccept: */*\r\nConnection: close\r\nAuthorization: Basic %s\r\n\r\n' % (
            self.mountpoint, encoded_credentials)
        self.ntrip_tcp_sock.sendall(server_request)

        while True:
            try:
                response = self.ntrip_tcp_sock.recv(10000)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                    continue
                else:
                    # a "real" error occurred
                    print(e)
                    return False
            else:
                #print(response)
                if 'ICY 200 OK' in response:
                    print('Successfully connected to NTRIP caster')
                    return True
                else:
                    print('Received unexpected response from caster:\n%s' % response)
                    return False

    def run(self):
        print('Starting NTRIP TCP socket thread')
        while not self.stop_event.isSet():

            if not self.connected_to_caster:
                if self.connect_to_ntrip_caster():
                    self.connected_to_caster = True
                else:
                    time.sleep(0.05)
                    continue

            # Receive RTCM messages from NTRIP caster and put in queue to send to GPS receiver
            try:
                ready_to_read, ready_to_write, in_error = select.select([self.ntrip_tcp_sock, ], [self.ntrip_tcp_sock, ], [], 5)
                #print ready_to_read
            except select.error:
                self.ntrip_tcp_sock.close()
                self.connected_to_caster = False
                print('Error calling select(): resetting connection to NTRIP caster')
                continue

            if len(ready_to_read) > 0:
                rtcm_msg = self.ntrip_tcp_sock.recv(100000)
                if len(rtcm_msg) > 0:
                    if ord(rtcm_msg[0]) == 0xD3:
                        rtcm_msg_len = 256 * ord(rtcm_msg[1]) + ord(rtcm_msg[2])
                        rtcm_msg_no = (256 * ord(rtcm_msg[3]) + ord(rtcm_msg[4])) / 16
                        #print('Received RTCM message %d with length %d' % (rtcm_msg_no, rtcm_msg_len))
                    else:
                        # print('Received ASCII message from server: %s' % str(rtcm_msg))
                        print('%d' % ord(rtcm_msg[0]))

                    rtcm_queue.put(rtcm_msg)
                    self.no_rtcm_data_count = 0

            # Get GPGGA messages from receive queue and send
            # to NTRIP server to keep connection alive
            if len(ready_to_write) > 0:
                try:
                    gga_msg = gga_queue.get_nowait()
                    #print('Sending new GGA message to NTRIP caster %s' % gga_msg)
                    self.ntrip_tcp_sock.sendall(gga_msg)
                    self.sent_gga = True
                except Queue.Empty:
                    pass

            if self.no_rtcm_data_count > 200:
                print('No RTCM messages for 10 seconds; resetting connection to NTRIP caster')
                self.ntrip_tcp_sock.close()
                self.connected_to_caster = False
                self.no_rtcm_data_count = 0

            if self.sent_gga:
                self.no_rtcm_data_count += 1

            time.sleep(0.05)

        print('Stopping NTRIP TCP socket thread')
        self.ntrip_tcp_sock.close()

    def stop(self):
        if(self.connected_to_caster):
            print("#######################\nTcp port will be closed \n#######################")
            self.ntrip_tcp_sock.close()
            self.connected_to_caster = False
        self.stop_event.set()


class ReceiverThread (threading.Thread):
    def __init__(self, com_port, com_baud):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.serial_socket = None
        self.com_port = com_port
        self.com_baud = com_baud
        self.com_parity = serial.PARITY_NONE
        self.com_stop = serial.STOPBITS_ONE
        self.com_bytesize = serial.EIGHTBITS
        #self.old_rtcm_msg = None

    def run(self):
        if(self.com_port and self.com_baud and self.com_parity and self.com_stop and self.com_bytesize):
			self.serial_socket = serial.Serial(port=self.com_port, baudrate=self.com_baud, parity=self.com_parity, 
								stopbits=self.com_stop, bytesize=self.com_bytesize)

        if(self.serial_socket.isOpen()):
            print("#######################\nCom port is already open, Closing \n#######################")
            self.serial_socket.close()

        self.serial_socket.open()

        print('Starting relay socket thread')
        while not self.stop_event.isSet():
            # Get RTCM messages from NTRIP TCP socket queue and send to GPS receiver over UDP
            try:
                rtcm_msg = rtcm_queue.get_nowait()
                if self.serial_socket and self.serial_socket.isOpen():
                    self.serial_socket.write(rtcm_msg)
                self.old_rtcm_msg = rtcm_msg
            except Queue.Empty:
                # Nothing in the RTCM message queue this time
                #if self.old_rtcm_msg is not None:
                #    self.serial_socket.write(self.old_rtcm_msg)
                pass

            time.sleep(0.05)

    def stop(self):
        if(self.serial_socket.isOpen()):
            print("#######################\nCom port will be closed \n#######################")
            self.serial_socket.close()
        self.stop_event.set()


def stop_threads(workers):
    for worker in workers:
        worker.stop()
        worker.join()


def start_threads(caster_ip, caster_port, mountpoint, username, password, com_port, com_baud):
    workers = [NtripSocketThread(caster_ip, caster_port, mountpoint, username, password), ReceiverThread(com_port, com_baud)]

    for worker in workers:
        worker.start()
    return workers


class RosInterface:
    def __init__(self):
        rospy.init_node('ntrip_client')

        self.gga_call_msg = None

        self.caster_ip = rospy.get_param('~caster_ip', default='')
        self.caster_port = rospy.get_param('~caster_port', default=0)
        self.mountpoint = rospy.get_param('~mountpoint', default='')
        self.username = rospy.get_param('~ntrip_username', default='')
        self.password = rospy.get_param('~ntrip_password', default='')
        #self.lateral = rospy.get_param('~lateral', default=50.830614)
        #self.longitudinal = rospy.get_param('~longitudinal', default=6.024385)
        #self.height = rospy.get_param('~height', default=300)
        self.map_origin = rospy.get_param('~map_origin', default=(50.8306145, 6.0243796, 0))

        self.com_port = rospy.get_param('~com_port', default="/dev/ttyUSB0")
        self.com_baud = rospy.get_param('~com_baud', default=19200)

        try:
            self.gnss_topic = rospy.get_param('~gnss_topic')
            rospy.Subscriber(self.gnss_topic, NavSatFix, self.recv_gga)
        except:
            pass

        self.gga_timer_now = rospy.Timer(rospy.Duration(0.01), self.gga_timer_cb, oneshot=True) # Uglyyyyyyy. Only because the timer below waits 5 seconds until it sends the gga for the first time
        self.gga_timer = rospy.Timer(rospy.Duration(5.0), self.gga_timer_cb)
        #rospy.Subscriber('gps/gga', String, self.recv_gga)
        
        self.workers = start_threads(self.caster_ip, self.caster_port, self.mountpoint, self.username, self.password, self.com_port, self.com_baud)

    def recv_gga(self, msg):
        self.gga_call_msg = msg #.data

    def gga_timer_cb(self, event):
        now = datetime.datetime.utcnow()
        hour = now.hour
        minute = now.minute
        second = now.second

        if self.gga_call_msg is not None:
            #gga_queue.put(self.gga_call_msg)
            position = (self.gga_call_msg.latitude, self.gga_call_msg.longitude, self.gga_call_msg.altitude)

            self.setPosition(position)

            utc_time = time.gmtime(self.gga_call_msg.header.stamp.secs)
            hour = utc_time[3]
            minute = utc_time[4]
            second = utc_time[5]

            self.gga_call_msg = None
        else:
            self.setPosition(self.map_origin)

        gga_msg= "GPGGA,%02d%02d%04.2f,%02d%011.8f,%1s,%03d%011.8f,%1s,1,05,0.19,+00400,M,%5.3f,M,," % (hour,minute,second,self.latDeg,self.latMin,self.flagN,self.lonDeg,self.lonMin,self.flagE,self.height)
        checksum = self.calcultateCheckSum(gga_msg)
        gga_msg_check = "$%s*%s\r\n" % (gga_msg, checksum)

        gga_queue.put(gga_msg_check)

    def on_shutdown(self):
        print('Shutting down')
        stop_threads(self.workers)

    def setPosition(self, map_origin):
        lat = map_origin[0]
        lon = map_origin[1]
        self.height = map_origin[2]
        self.flagN="N"
        self.flagE="E"
        if lon>180:
            lon=(lon-360)*-1
            self.flagE="W"
        elif (lon<0 and lon>= -180):
            lon=lon*-1
            self.flagE="W"
        elif lon<-180:
            lon=lon+360
            self.flagE="E"
        else:
            self.lon=lon
        if lat<0:
            lat=lat*-1
            self.flagN="S"
        self.lonDeg=int(lon)
        self.latDeg=int(lat)
        self.lonMin=(lon-self.lonDeg)*60
        self.latMin=(lat-self.latDeg)*60

    def calcultateCheckSum(self, stringToCheck):
        xsum_calc = 0
        for char in stringToCheck:
            xsum_calc = xsum_calc ^ ord(char)
        return "%02X" % xsum_calc

def main():
    ros_interface = RosInterface()
    rospy.on_shutdown(ros_interface.on_shutdown)

    rospy.spin()

if __name__ == '__main__':
    main()
