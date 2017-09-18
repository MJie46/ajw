from django.http import HttpResponse
from django.shortcuts import render

from people.models import Person
from people.models import TrainResult
import binascii
import sqlite3
from PIL import Image
import numpy as np
import io
import re
import json

'''
% s字符串(采用str()的显示)
% r字符串(采用repr()的显示)
% c单个字符
% b二进制整数
% d十进制整数
% i十进制整数
% o八进制整数
% x十六进制整数
% e指数(基底写为e)
% E指数(基底写为E)
% f浮点数
% F浮点数，与上相同
% g指数(e)或浮点数(根据显示长度)
% G指数(E)或浮点数(根据显示长度)
% % 字符
"%"
'''
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        return bytes_or_str.encode('utf-8')
    # python3中接受bytes和str, 并总是返回bytes：
    return bytes_or_str

def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.decode('utf-8')
    # python3中接受bytes和str, 并总是返回str：
    return bytes_or_str

def isNumber(str):
    value = re.compile(r'[0-9]{0,}')
    result = value.match(str)
    print (result)
    if result:
        return True
    else:
        return False

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

# 数据库操作
def testdb(request):
    test1 = Person(name='runoob')
    test1.save()
    return HttpResponse("<p>数据添加成功！</p>")

def insertGuestData(request):

    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()

    f = open('./guestlist.txt','r')
    tableid = 0
    for name in f.readlines():
        name = name.replace('\n','')
        # print (name)
        # print (len(name))
        if is_number(name):
            tableid = name
            # print (tableid)
            continue
        sql = "INSERT INTO guest (name,tableid) VALUES (?,?)"
        value = (name, tableid)
        conn.execute(sql,value)
        conn.commit()


    # print ('select')
    # cursor = conn.execute("SELECT * from guest")
    # ctn = 0
    # for row in cursor:
    #     ctn += 1
    #     data = row[1]
    #     print (data)


    table1 = []


    # cursor.execute("create table guest (id integer PRIMARY KEY AUTOINCREMENT,name text,phone text,tableid integer)")
    # ctn = 0
    # for row in cursor:
    #     ctn += 1
    #     data = row[1]

def index(request):
    # p1 = Person(name='sdflsdf',age=12)
    # p1.save()
    # list = Person.objects.all()
    # print (list)


    # f = open('./1.png','rb')
    # data = f.read()
    # datastr = data.decode()
    # print (type(datastr))

    # # Converts np.array to TEXT when inserting
    # sqlite3.register_adapter(np.ndarray, adapt_array)
    # # Converts TEXT to np.array when selecting
    # sqlite3.register_converter("array", convert_array)
    # # con = sqlite3.connect("db.sqlite3", detect_types=sqlite3.PARSE_DECLTYPES)
    # cursor.execute("create table test (id integer PRIMARY KEY AUTOINCREMENT,arr array)")

    # x = np.random.normal(size=[32,32,3])
    # image = Image.open('./1.png')
    # x = np.asarray(image)
    # y = Image.fromarray(x)
    # cursor.execute("insert into test (arr) values (?)", (x,))

    # t1 = TrainResult(step=1,resultId=1,image=data,width=64,height=64)
    # t1.save()
    # sql = "INSERT INTO people_trainresult (step,resultId,image,width,height) VALUES (?,?,?,?,?)"
    # value = (1, 1, data, 64, 64)
    # cursor = conn.execute(sql,value)
    # conn.commit()

    print ('select')
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.execute("SELECT * from test")
    ctn = 0
    for row in cursor:
        ctn += 1
        data = row[1]
        data = np.loadtxt(data)
        data = np.array(data,dtype='uint8')
        data = np.reshape(data,[64,64,3])
        image = Image.fromarray(data)
        # image.save('static/image/p.png')
        # image.show()


    cursor.close()
    print (ctn)
    images1 = ['%d.png'%i for i in range(64)]
    images2 = ['%d.png' % (i+64) for i in range(64)]
    # images1 = []
    # images2 = []
    print(images1)
    return render(request, 'hello.html', {'images1':images1,'images2':images2})
    # return HttpResponse("Hello world !!! ")

def seatSearch(request):
    print ('initializ seat seatSearch')

    return render(request, 'seatSearch.html', {})

def seatDetail(request):

    print ('initializ seat seatDetail')
    request.encoding='utf-8'

    params = request.GET
    name = request.GET['name']
    message = '你搜索的内容为: ' + name
    print (message)

    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    sql = "SELECT * FROM guest WHERE NAME = ?"
    cursor.execute(sql,(name,))
    result = cursor.fetchall()
    tableid = 0
    for row in result:
        tableid = row[3]

    sql = "SELECT * FROM guest"
    cursor.execute(sql,())
    result = cursor.fetchall()
    guestall = {}
    lastindex = -2
    ctn = -1
    for row in result:
        guestrow = None
        tid = row[3]
        if tid not in guestall.keys():
            guestrow = []
            guestall[tid] = guestrow
        else:
            guestrow = guestall[tid]
        guestrow.append(row)

    # print (guestall)
    # print (len(guestall))


    return render(request, 'seatDetail.html', {'tableid':tableid,'guestall':json.dumps(guestall)})


def testanimte(request):

    return render(request, 'testanimte.html', {})

