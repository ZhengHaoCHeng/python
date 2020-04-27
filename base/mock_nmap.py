#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Python3 模拟实现nmap的TCP全连接扫描和SYN扫描
多进程用于进程多主机扫描，多线程用于对主机的端口扫描

使用 SYN 扫描请使用管理员权限

scapy 现在改名为了 kamene

参考文章：
https://blog.csdn.net/fageweiketang/article/details/83752750
https://www.cnblogs.com/Chinori/archive/2019/09/21/11560211.html
https://www.jianshu.com/p/42e5a9bb7268
https://zhuanlan.zhihu.com/p/22906698
----------
Usage:
	python3 mock_nmap.py -T tcp -W 3 -H 192.168.1.1
	python3 mock_nmap.py -T tcp -H 192.168.1.1,192.168.1.11
	python3 mock_nmap.py -T tcp -H 192.168.1.0/24
	sudo python3 mock_nmap.py -T syn -H 192.168.1.1
"""
import socket
import argparse
import ipaddress
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from scapy.all import *
import time
import os


def syn_scan(host, port):
	package = IP(dst=host)/TCP(dport=port, flags="S")
	try:
		res = sr1(package, timeout=1, verbose=0)
	except OSError:                      # 需要管理员权限
		print('[*] Operation not permitted!')
		print('[*] Only usable as Root/Administrator!')
		os._exit(0)
	if res:
		if str(res['TCP'].flags) == 'SA':
			# print('[+] %d open' % port)
			return '[+] %d open' % port
		else:
			# print('[-] %d close' % port)
			return None
	else:
		# print('[-] %d close' % port)
		return None


def tcp_scan(host, port):
	socket.setdefaulttimeout(1)
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	result = s.connect_ex((host, port))  # result 为 0 说明开放
	s.close()
	if result == 0:
		# print('[+] %d open' % port)
		return '[+] %d open' % port
	else:
		# print('[-] %d close' % port)
		return None


def threadpool_scan_port(host, func_name, ports, process_lock):
	result = {'report':[], 'open_num':0}
	func =  globals().get(func_name)

	with ThreadPoolExecutor(max_workers=5) as executor:            # 每个进程再开5个线程扫描端口
		all_task = [executor.submit(func, host, port) for port in ports]

	for task in all_task:
		res = task.result()
		result['report'].append(res)

	result['report'] = list(filter(lambda x:x, result['report']))  # 只保留开放的端口
	result['open_num'] = len(result['report'])                     # 开放的端口数量
	print_result(host, result, process_lock)                       # 打印扫描结果，用锁防止两个进程同时打印


def print_result(host, result, process_lock):
	process_lock.acquire()
	print('scan report for %s ' % host)
	for r in result['report']:
		print(r)
	print('shown %d opened ports' % result['open_num'], end='\n\n')
	process_lock.release()


def main():
	print('Starting Mock_Nmap at %s' % time.ctime())

	parser = argparse.ArgumentParser()
	parser.add_argument('-T', '--type', type=str, choices=['tcp', 'syn'], default='tcp', help='tcp指TCP全连接扫描；syn指TCP半连接扫描，需要管理员权限')
	parser.add_argument('-H', '--hosts', type=str, required=True, help='要扫描的主机，支持单个IP或CIDR地址块，多IP请以英文“,”分隔')
	parser.add_argument('-W', '--workers', type=int, default=2, help='用于扫描不同的主机的进程数')
	args = parser.parse_args()

	ports = [21, 22, 80, 135, 443, 445, 1433, 3306, 3389, 8080]
	
	if args.type == 'tcp':
		scan_method = 'tcp_scan'
	else:
		scan_method = 'syn_scan'
	scan_hosts = args.hosts
	if ',' in scan_hosts:
		scan_hosts = scan_hosts.split(',')
	elif '/' in scan_hosts:
		scan_hosts = [h for h in ipaddress.ip_network(scan_hosts).hosts()]
	else:
		scan_hosts = [scan_hosts]

	worker_num = args.workers            # 进程数
	process_lock = Manager().Lock()      # 进程锁

	start_time = time.time()
	try:
		with ProcessPoolExecutor(max_workers=worker_num) as executor:
			all_task = [executor.submit(threadpool_scan_port, str(host), scan_method, ports, process_lock) for host in scan_hosts]
	except KeyboardInterrupt:            # Ctrl+C 退出
		os._exit(0)
	end_time = time.time()
	print('Mock_Nmap done scanned in %d seconds' % (end_time-start_time))


if __name__ == '__main__':
	main()
