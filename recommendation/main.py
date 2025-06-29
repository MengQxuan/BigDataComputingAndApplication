import os
import time
import psutil
from memory_profiler import memory_usage
from Analyze import AnalyzeData
from Train import train

def get_memory_usage():
    """
    获取当前进程的内存使用情况（RSS），单位为 MB。
    """
    pid = os.getpid()
    p = psutil.Process(pid)
    memory_usage = p.memory_info().rss / 1024 / 1024  # MB
    return memory_usage

def monitor_training():
    """
    使用 memory_profiler 监测训练过程中的内存消耗峰值，并计算运行时间。
    """
    start_time = time.perf_counter()  # 使用更高分辨率的计时方法
    start_memory = get_memory_usage()

    # 监测内存峰值并获取返回值
    mem_usage, _ = memory_usage((train,), interval=0.1, retval=True)

    end_time = time.perf_counter()
    end_memory = get_memory_usage()

    max_memory = max(mem_usage)  # 获取内存峰值

    print(f"共花费 {end_time - start_time:.2f} 秒")
    print(f'内存消耗: {end_memory:.2f} MB')
    print(f'内存峰值: {max_memory:.2f} MB')

def main():
    """
    主程序函数，执行数据分析、训练，并输出运行时间和内存消耗。
    """
    AnalyzeData()  # 数据分析处理
    monitor_training()

if __name__ == '__main__':
    main()
