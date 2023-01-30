 #!/usr/bin/env python

import socket
import os
import time
import sys
import mysql.connector
import binascii

BUFFER_SIZE = 4096
FORMAT = "utf-8"

'''change db credentials  as needed'''
check_conn = mysql.connector.connect(host="localhost",user="root",password="")
cursor = check_conn.cursor()
cursor.execute("CREATE DATABASE IF NOT EXISTS test_db")
cursor.close()


connection = mysql.connector.connect(host= "localhost", user="root", password="", database="test_db")
cursor = connection.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS test_img_db ( id INT(100) NOT NULL AUTO_INCREMENT, location VARCHAR(50) NOT NULL , date_time VARCHAR(50) NOT NULL , fire INT(10) NOT NULL , smoke INT(10) NOT NULL ,raise_alert INT(11) NOT NULL ,image BLOB NOT NULL , PRIMARY KEY (`id`)) ENGINE = InnoDB;")
cursor.close()
connection.close()

#Staring a TCP socket.
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#Bind the IP and PORT to the server.
server.bind(('169.254.13.181', 3333))

#Server is listening, i.e., server is now waiting for the client to connected.
server.listen()

while True:
    
    #image transfer
    client_socket, client_address = server.accept()
    
    image_filename = client_socket.recv(BUFFER_SIZE).decode(FORMAT)
    file = open(image_filename, "wb")
    client_socket.send("Filename received.".encode(FORMAT))
    
    image_content = bytes("",'utf-8')
    image_data = client_socket.recv(BUFFER_SIZE)
 
    while image_data:
        image_content = image_content+image_data
        image_data = client_socket.recv(BUFFER_SIZE)
    file.write(image_content)

    '''USE image_content to store in db'''
    
    file.close()
    client_socket.close()
    
    #text file transfer
    
    conn, addr = server.accept()
    filename = conn.recv(BUFFER_SIZE).decode(FORMAT)
    file = open(filename, "w")
    conn.send("Filename received.".encode(FORMAT))

    data = conn.recv(BUFFER_SIZE).decode(FORMAT)
    file.write(data)
    conn.send("File data received".encode(FORMAT))

    file.close()
    conn.close()

    #prepare data to upload on database
    image_binary_data = None
    file = open(image_filename,'rb')
    image_binary_data = file.read()
    file.close()
    
    with open(filename,'r') as file:
        temp=0
        for line in file:
            for word in line.split():
                #print(word)
                if word == "location":
                    temp=1
                elif word == "date_time":
                    temp=2
                elif word == "fire":
                    temp = 3
                elif word == "smoke":
                    temp = 4
                elif word == "raise_alert":
                    temp = 5
                else:
                    if temp == 1:
                        location = word
                    elif temp == 2:
                        date_time = word
                    elif temp == 3:
                        fire = word
                    elif temp == 4:
                        smoke = word
                    elif temp == 5:
                        raise_alert = word
                        alert_history = word
    file.close()

    #upload data on database
    connection = mysql.connector.connect(host= "localhost", user="root", password="", database="test_db")
    cursor = connection.cursor()
    InsertQuery = """ INSERT into test_img_db (id, location,date_time,fire,smoke,raise_alert,alert_history,image) values (%s,%s,%s,%s,%s,%s,%s,%s)"""
    value = ('+', location, date_time, fire, smoke, raise_alert, alert_history, image_binary_data)
    cursor.execute(InsertQuery,value)
    #cursor.execute("INSERT INTO test_img_db (id, location, date_time, fire, smoke, raise_alert ,image) VALUES ('+', '"+location+"', '"+date_time+"', '"+fire+"', '"+smoke+"', '"+raise_alert+"', %s)",(image_binary_data))
    connection.commit()
    cursor.close()
    
    '''cursor = connection.cursor()
    selectQuery = "Select * From test_img_db where date_time = %s"
    value = (date_time,)
    cursor.execute(selectQuery,value)
    result = cursor.fetchmany(size = 6)
    file = open(f'image_from_db.jpeg','wb')
    for row in result:
        file.write(row[8])
    cursor.close()
    file.close()
    connection.close()
    print("Data upload Successful")'''

    
    os.remove(image_filename)
    os.remove(filename)
    


sys.exit()
