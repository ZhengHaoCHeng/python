# -*- coding:utf-8 -*-
from multiprocessing import Process, Pool
from time import ctime, sleep, time
from concurrent.futures import ProcessPoolExecutor


def loop(nloop, nsec):
	print("start loop", nloop, "at:", ctime())
	sleep(nsec)
	print("loop", nloop, "done at:", ctime())


def with_join():
	# 不使用循环启动
	# p1 = Process(target=loop, args=(1, 4))
	# p2 = Process(target=loop, args=(2, 3))
	# p1.start()
	# p2.start()
	# p1.join()
	# p2.join()
	# 使用循环启动
	process_list = []
	for i, j in zip([1,2],[4,3]):
		process_list.append(Process(target=loop, args=(i, j)))
	for p in process_list:
		p.start()
	for p in process_list:  # 多进程不一定要使用join()
		p.join()            # 使用了join()的话主进程等待全部子进程完成后才继续往下执行


def no_join():
	process_list = []
	for i, j in zip([1,2],[4,3]):
		process_list.append(Process(target=loop, args=(i, j)))
	for p in process_list:
		p.start()


def pool():
	# 设置多进程数
	pool = Pool(processes=3)
	for i, j in zip([1,2],[4,3]):
		# 维持执行的进程总数为3，当一个进程执行完毕后会添加新的进程进去
		pool.apply_async(loop, args=(i, j))
	pool.close()
	# 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool，join函数等待所有子进程结束
	pool.join()


def new_pool():
	with ProcessPoolExecutor(max_workers=3) as executor:
		all_task = [executor.submit(loop, i, j) for i, j in zip([1,2],[4,3])]


if __name__=="__main__":
	print("使用join()")
	start = time()
	with_join()
	print("time: ", time()-start, end="\n\n")
	
	print("不使用join()")
	start = time()
	no_join()
	print("time: ", time()-start)

	sleep(4)
	print("")
	print("使用Pool")
	start = time()
	pool()
	print("time: ", time()-start)

	print("")
	print("使用ProcessPoolExecutor")
	start = time()
	new_pool()
	print("time: ", time()-start)
