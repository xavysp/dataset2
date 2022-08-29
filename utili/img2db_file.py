"""

"""
import os
import sys
import psycopg2
# import argparse

def readImage(img_path):
    try:
        fin = open(img_path, "rb")
        img = fin.read()
        return img

    except (Exception, IOError) as e:
        print ("Error %d: %s" % (e.args[0],e.args[1]))
        sys.exit(1)

    finally:
        if fin:
            fin.close()

def img2db(img_list, igm_dir):
    try:
        con = psycopg2.connect(database="testdb", user="myprojectuser")
        cur = con.cursor()
        data = readImage()
        binary = psycopg2.Binary(data)
        cur.execute("INSERT INTO images(id, data) VALUES (1, %s)", (binary,) )
        con.commit()
    except (Exception, psycopg2.DatabaseError) as error:

        if con:
            con.rollback()
        print('Error %s' % error)
        sys.exit(1)
    finally:
        if con:
            con.close()
