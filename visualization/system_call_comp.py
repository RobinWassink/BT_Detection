import csv
import os
import re
import sys
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

syscall_list=["llseek", "_newselect", "_sysctl", "accept", "accept4", "access", "acct", "add_key", "adjtimex", "alarm", "alloc_hugepages", "bdflush", "bind", "bpf", "brk", "cacheflush", "capget", "capset", "chdir", "chmod", "chown", "chown32", "chroot", "clock_adjtime", "clock_getres", "clock_gettime", "clock_nanosleep", "clock_settime", "clone", "close", "connect", "copy_file_range", "creat", "create_module", "delete_module", "dup", "dup2", "dup3", "epoll_create", "epoll_create1", "epoll_ctl", "epoll_pwait", "epoll_wait", "eventfd", "eventfd2", "execve", "execveat", "exit", "exit_group", "faccessat", "fadvise64", "fadvise64_64", "fallocate", "fanotify_init", "fanotify_mark", "fchdir", "fchmod", "fchmodat", "fchown", "fchown32", "fchownat", "fcntl", "fcntl64", "fdatasync", "fgetxattr", "finit_module", "flistxattr", "flock", "fork", "free_hugepages", "fremovexattr", "fsetxattr", "fstat", "fstat64", "fstatat64", "fstatfs", "fstatfs64", "fsync", "ftruncate", "ftruncate64", "futex", "futimesat", "get_kernel_syms", "get_mempolicy", "get_robust_list", "get_thread_area", "getcpu", "getcwd", "getdents", "getdents64", "getegid", "getegid32", "geteuid", "geteuid32", "getgid", "getgid32", "getgroups", "getgroups32", "getitimer", "getpeername", "getpagesize", "getpgid", "getpgrp", "getpid", "getppid", "getpriority", "getrandom", "getresgid", "getresgid32", "getresuid", "getresuid32", "getrlimit", "getrusage", "getsid", "getsockname", "getsockopt", "gettid", "gettimeofday", "getuid", "getuid32", "getxattr", "init_module", "inotify_add_watch", "inotify_init", "inotify_rm_watch", "io_cancel", "io_destroy", "io_getevents", "io_setup", "io_submit", "ioctl", "ioperm", "iopl", "ioprio_get", "ioprio_set", "ipc", "kcmp", "kern_features", "kexec_file_load", "kexec_load", "keyctl", "kill", "lchown", "lchown32", "lgetxattr", "link", "linkat", "listen", "listxattr", "llistxattr", "lookup_dcookie", "lremovexattr", "lseek", "lsetxattr", "lstat", "lstat64", "madvise", "mbind", "memfd_create", "migrate_pages", "mincore", "mkdir", "mkdirat", "mknod", "mknodat", "mlock", "mlock2", "mlockall", "mmap", "mmap2", "modify_ldt", "mount", "move_pages", "mprotect", "mq_getsetattr", "mq_notify", "mq_open", "mq_timedreceive", "mq_timedsend", "mq_unlink", "mremap", "msgctl", "msgget", "msgrcv", "msgsnd", "msync", "munlock", "munlockall", "munmap", "name_to_handle_at", "nanosleep", "nfsservctl", "nice", "oldfstat", "oldlstat", "oldolduname", "oldstat", "olduname", "open", "open_by_handle_at", "openat", "pause", "pciconfig_iobase", "pciconfig_read", "pciconfig_write", "perf_event_open", "personality", "perfctr", "perfmonctl", "pipe", "pipe2", "pivot_root", "pkey_alloc", "pkey_free", "pkey_mprotect", "poll", "ppc_rtas", "ppc_swapcontext", "ppoll", "prctl", "pread64", "preadv", "preadv2", "prlimit64", "process_vm_readv", "process_vm_writev", "pselect6", "ptrace", "pwrite64", "pwritev", "pwritev2", "query_module", "quotactl", "read", "readahead", "readdir", "readlink", "readlinkat", "readv", "reboot", "recv", "recvfrom", "recvmsg", "recvmmsg", "remap_file_pages", "removexattr", "rename", "renameat", "renameat2", "request_key", "restart_syscall", "rmdir", "rt_sigaction", "rt_sigpending", "rt_sigprocmask", "rt_sigqueueinfo", "rt_sigreturn", "rt_sigsuspend", "rt_sigtimedwait", "rt_tgsigqueueinfo", "s390_runtime_instr", "s390_pci_mmio_read", "s390_pci_mmio_write", "sched_get_priority_max", "sched_get_priority_min", "sched_getaffinity", "sched_getattr", "sched_getparam", "sched_getscheduler", "sched_rr_get_interval", "sched_setaffinity", "sched_setattr", "sched_setparam", "sched_setscheduler", "sched_yield", "seccomp", "select", "semctl", "semget", "semop", "semtimedop", "send", "sendfile", "sendfile64", "sendmmsg", "sendmsg", "sendto", "set_mempolicy", "set_robust_list", "set_thread_area", "set_tid_address", "setdomainname", "setfsgid", "setfsgid32", "setfsuid", "setfsuid32", "setgid", "setgid32", "setgroups", "setgroups32", "sethostname", "setitimer", "setns", "setpgid", "setpriority", "setregid", "setregid32", "setresgid", "setresgid32", "setresuid", "setresuid32", "setreuid", "setreuid32", "setrlimit", "setsid", "setsockopt", "settimeofday", "setuid", "setuid32", "setup", "setxattr", "sgetmask", "shmat", "shmctl", "shmdt", "shmget", "shutdown", "sigaction", "sigaltstack", "signal", "signalfd", "signalfd4", "sigpending", "sigprocmask", "sigreturn", "sigsuspend", "socket", "socketcall", "socketpair", "splice", "spu_create", "spu_run", "ssetmask", "stat", "stat64", "statfs", "statfs64", "stime", "subpage_prot", "swapoff", "swapon", "symlink", "symlinkat", "sync", "sync_file_range", "sync_file_range2", "syncfs", "sysfs", "sysinfo", "syslog", "tee", "tgkill", "time", "timer_create", "timer_delete", "timer_getoverrun", "timer_gettime", "timer_settime", "timerfd_create", "timerfd_gettime", "timerfd_settime", "times", "tkill", "truncate", "truncate64", "ugetrlimit", "umask", "umount", "umount2", "uname", "unlink", "unlinkat", "unshare", "uselib", "ustat", "userfaultfd", "utime", "utimensat", "utimes", "utrap_install", "vfork", "vhangup", "vm86old", "vm86", "vmsplice", "wait4", "waitid", "waitpid", "write", "writev"]

# first version with pandas dataframe, takes ~ 15x longer 
def countSysCalls(folder, mode, syscalls):
    print("Started for mode: %s"%mode)
    t1 = time.time()
    df = pd.DataFrame(0, index=range(0,300), columns=syscalls)
    regex = re.compile('.*(%s).*'%mode)
    for root, dirs, files in os.walk(folder):
        for file in files:
            if regex.match(file):
                fis = file.split('_')
                step = fis[1]
                with open(folder+"/"+file, "r") as systemCallsFile:
                    csvReader = csv.reader(systemCallsFile, delimiter=",")
                    # next(csvReader)
                    for row in csvReader:
                        df.at[int(step), row[2]] += 1
    t2 = time.time()
    print("This took ", t2 - t1, "seconds")
    return df

# counting the syscalls with dictionaries, faster
def countSysCalls_dic(folder, mode, syscalls):
    print("Started for mode: %s"%mode)
    t1 = time.time()
    sysCalls = {}
    regex = re.compile('.*(%s).*'%mode)
    for root, dirs, files in os.walk(folder):
        for file in files:
            if regex.match(file):
                fis = file.split('_')
                step = fis[1]
                sysCalls[step] = dict.fromkeys(syscalls, 0)
                with open(folder+"/"+file, "r") as systemCallsFile:
                    csvReader = csv.reader(systemCallsFile, delimiter=",")
                    # next(csvReader)
                    for row in csvReader:
                        if row[2] in syscall_list:
                            sysCalls[step][row[2]] += 1
                            
    t2 = time.time()
    print("This took ", t2 - t1, "seconds")
    return sysCalls

def main(argv):
    folder = argv[0]
    dirname = os.path.dirname(__file__)
    raw_folder = os.path.abspath(os.path.join(dirname, "../data/" + folder +"/raw"))
    visualization_path = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/visualization/"))
    modes = ["normal", "repeat", "mimic", "confusion", "noise", "spoof", "freeze", "delay"]
    
    if os.path.exists(os.path.abspath(os.path.join(visualization_path, "df.pk"))):
        print("Found df.pk file")
        loc=open(os.path.abspath(os.path.join(visualization_path, "df.pk")),'rb')
        df = pickle.load(loc)
    else:
        if os.path.exists(os.path.abspath(os.path.join(visualization_path, "syscalls_total.csv"))):
            syscalls_total = pd.read_csv(os.path.abspath(os.path.join(visualization_path, "syscalls_total.csv")), sep=',')
        else: 
            print("no total syscalls table found!")
            return 0
        
        syscalls = list(syscalls_total.columns.values)
        syscalls_cleaned = [x for x in syscalls if x in set(syscall_list)]
        sysCallsDic = {}
        for mode in modes:
            sysCallsDic[mode] = countSysCalls_dic(raw_folder, mode, syscalls_cleaned)

        df = pd.DataFrame.from_dict({(i,j): sysCallsDic[i][j] 
                            for i in sysCallsDic.keys() 
                            for j in sysCallsDic[i].keys()},
                        orient='index')

        pickle.dump(df, open(os.path.abspath(os.path.join(visualization_path, "df.pk")), "wb"))

    # print(df)
    # print(df.shape)
    # print(df.loc["normal"]["ioctl"])
    # print(df["ioctl"])
    # print(df.columns)

    interesting_syscalls = ["ioctl", "poll", "timerfd_settime", "close", "open", "write"]

    for syscall in df.columns:
        if (syscall in interesting_syscalls):
            plt.rcParams["figure.figsize"] = (15,3) 
            plt.figure()
            ymin, ymax = 30000, 0
            for mode in modes:
                data = df.loc[mode][syscall]
                data.index = data.index.map(int)
                data = data.sort_index()
                plt.plot(data, label=mode)
                if data.quantile(.05) < ymin: ymin = data.quantile(.05) 
                if max(data) > ymax: ymax = max(data)
            if (ymin < 10): ymin = -15
            plt.ylim(0.95*ymin, 1.05*ymax)
            plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
            plt.title(syscall, fontsize=20, weight="bold")
            plt.savefig(os.path.abspath(os.path.join(visualization_path, syscall+"_evolution.png")), bbox_inches="tight")
            #plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
