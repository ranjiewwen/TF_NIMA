#!/usr/bin/env python
# -*- coding:utf-8 -*-

from queue import Queue

# Queue是python标准库中的线程安全的队列（FIFO）实现,提供了一个适用于多线程编程的先进先出的数据结构，即队列，用来在生产者和消费者线程之间的信息传递
def test_queue():

    q=Queue(10)
    for i in range(5):
        q.put(i)
    while not q.empty():
        print(q.get())

def test_LifoQueue():
    import queue
    # queue.LifoQueue() #后进先出->堆栈
    q = queue.LifoQueue(3)
    q.put(1)
    q.put(2)
    q.put(3)
    print(q.get())
    print(q.get())
    print(q.get())

def test_PriorityQueue():
    import queue
    # queue.PriorityQueue() #优先级
    q = queue.PriorityQueue(3)  # 优先级,优先级用数字表示,数字越小优先级越高
    q.put((10, 'a'))
    q.put((-1, 'b'))
    q.put((100, 'c'))
    print(q.get())
    print(q.get())
    print(q.get())


# Python queue队列，实现并发，在网站多线程推荐最后也一个例子,比这货简单，但是不够规范

from queue import Queue  # Queue在3.x中改成了queue
import random
import threading
import time
from threading import Thread

class Producer(threading.Thread):
    """
    Producer thread 制作线程
    """
    def __init__(self, t_name, queue):  # 传入线程名、实例化队列
        threading.Thread.__init__(self, name=t_name)  # t_name即是threadName
        self.data = queue

    """
    run方法 和start方法:
    它们都是从Thread继承而来的，run()方法将在线程开启后执行，
    可以把相关的逻辑写到run方法中（通常把run方法称为活动[Activity]）；
    start()方法用于启动线程。
    """

    def run(self):
        for i in range(5):  # 生成0-4五条队列
            print("%s: %s is producing %d to the queue!" % (time.ctime(), self.getName(), i))  # 当前时间t生成编号d并加入队列
            self.data.put(i)  # 写入队列编号
            time.sleep(random.randrange(10) / 5)  # 随机休息一会
        print("%s: %s producing finished!" % (time.ctime(), self.getName))  # 编号d队列完成制作


class Consumer(threading.Thread):
    """
    Consumer thread 消费线程，感觉来源于COOKBOOK
    """
    def __init__(self, t_name, queue):
        threading.Thread.__init__(self, name=t_name)
        self.data = queue

    def run(self):
        for i in range(5):
            val = self.data.get()
            print("%s: %s is consuming. %d in the queue is consumed!" % (time.ctime(), self.getName(), val))  # 编号d队列已经被消费
            time.sleep(random.randrange(10))
        print("%s: %s consuming finished!" % (time.ctime(), self.getName()))  # 编号d队列完成消费


def main():
    """
    Main thread 主线程
    """
    queue = Queue()  # 队列实例化
    producer = Producer('Pro.', queue)  # 调用对象，并传如参数线程名、实例化队列
    consumer = Consumer('Con.', queue)  # 同上，在制造的同时进行消费
    producer.start()  # 开始制造
    consumer.start()  # 开始消费
    """
    join（）的作用是，在子线程完成运行之前，这个子线程的父线程将一直被阻塞。
　 　join()方法的位置是在for循环外的，也就是说必须等待for循环里的两个进程都结束后，才去执行主进程。
    """
    producer.join()
    consumer.join()
    print('All threads terminate!')


# 进程池 ,Pool中是有return的
import multiprocessing as mp

def job(x):
    return x ** 2

def multiprocess():
    pool = mp.Pool()  # 默认是有几个核就用几个，可以自己设置processes = ？
    res = pool.map(job, range(10))  # 可以放入可迭代对象，自动分配进程
    print(res)

    res = pool.apply_async(job, (2,))  # 一次只能在一个进程里计算，要达到map的效果，要迭代
    print(res.get())

    multi_res = [pool.apply_async(job, (i,)) for i in range(10)]  # 迭代器
    print([res.get() for res in multi_res])

# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
# 4
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

if __name__=="__main__":

    multiprocess()


    test_queue()

    print("=====后进先出=====")
    test_LifoQueue()

    print("=====优先级======")
    test_PriorityQueue()

    main()
