# -*- coding:utf-8 -*-
import threading
from time import sleep, ctime, time
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor


def loop(nloop, nsec):
	print("start loop", nloop, "at:", ctime())
	sleep(nsec)
	print("loop", nloop, "done at:", ctime())


def simple_threading():
	threads = []
	
	for i, j in zip([1,2],[4,3]):
		t = threading.Thread(target=loop, args=(i, j))
		threads.append(t)
	# 线程开始执行
	for t in threads:
		t.start()
	# 等待所有线程执行完成
	for t in threads:
		t.join()


def pool():
	pool = ThreadPool(processes=3)
	for i, j in zip([1,2],[4,3]):
		pool.apply_async(loop, args=(i, j))
	pool.close()
	pool.join()


def new_pool():
	with ThreadPoolExecutor(max_workers=3) as executor:
		all_task = [executor.submit(loop, i, j) for i, j in zip([1,2],[4,3])]


lock = threading.Lock()
def update(i): 
	lock.acquire()
	# 余额.txt 里面就 1000
	f = open("余额.txt", "r")
	money = int(f.readline().strip())
	f = open("余额.txt", "w")
	money = money - i
	f.write(str(money))
	lock.release()


def main():
	# 存入1000块
	with open("余额.txt", "w") as f:
		f.write(str(1000))
	thread_list = []
	for i in range(10):
		t = threading.Thread(target=update, args=(i,))
		thread_list.append(t)
	for t in thread_list:
		t.start()
	for t in thread_list:
		t.join()


if __name__ == "__main__":
	print("不使用线程池")
	start = time()
	simple_threading()    
	print("time: ", time()-start)

	print("")
	print("使用ThreadPool")
	start = time()
	pool()
	print("time: ", time()-start)

	print("")
	print("使用ThreadPoolExecutor")
	start = time()
	new_pool()
	print("time: ", time()-start)

	print("")
	print("使用线程锁防止数据抢占等问题")
	start = time()
	main()
	print("time: ", time()-start)
