import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os,sys
import tqdm
import pickle
import time
import numpy as np
import csv

max_n_gram = 4
syscall_list=["llseek", "_newselect", "_sysctl", "accept", "accept4", "access", "acct", "add_key", "adjtimex", "alarm", "alloc_hugepages", "bdflush", "bind", "bpf", "brk", "cacheflush", "capget", "capset", "chdir", "chmod", "chown", "chown32", "chroot", "clock_adjtime", "clock_getres", "clock_gettime", "clock_nanosleep", "clock_settime", "clone", "close", "connect", "copy_file_range", "creat", "create_module", "delete_module", "dup", "dup2", "dup3", "epoll_create", "epoll_create1", "epoll_ctl", "epoll_pwait", "epoll_wait", "eventfd", "eventfd2", "execve", "execveat", "exit", "exit_group", "faccessat", "fadvise64", "fadvise64_64", "fallocate", "fanotify_init", "fanotify_mark", "fchdir", "fchmod", "fchmodat", "fchown", "fchown32", "fchownat", "fcntl", "fcntl64", "fdatasync", "fgetxattr", "finit_module", "flistxattr", "flock", "fork", "free_hugepages", "fremovexattr", "fsetxattr", "fstat", "fstat64", "fstatat64", "fstatfs", "fstatfs64", "fsync", "ftruncate", "ftruncate64", "futex", "futimesat", "get_kernel_syms", "get_mempolicy", "get_robust_list", "get_thread_area", "getcpu", "getcwd", "getdents", "getdents64", "getegid", "getegid32", "geteuid", "geteuid32", "getgid", "getgid32", "getgroups", "getgroups32", "getitimer", "getpeername", "getpagesize", "getpgid", "getpgrp", "getpid", "getppid", "getpriority", "getrandom", "getresgid", "getresgid32", "getresuid", "getresuid32", "getrlimit", "getrusage", "getsid", "getsockname", "getsockopt", "gettid", "gettimeofday", "getuid", "getuid32", "getxattr", "init_module", "inotify_add_watch", "inotify_init", "inotify_rm_watch", "io_cancel", "io_destroy", "io_getevents", "io_setup", "io_submit", "ioctl", "ioperm", "iopl", "ioprio_get", "ioprio_set", "ipc", "kcmp", "kern_features", "kexec_file_load", "kexec_load", "keyctl", "kill", "lchown", "lchown32", "lgetxattr", "link", "linkat", "listen", "listxattr", "llistxattr", "lookup_dcookie", "lremovexattr", "lseek", "lsetxattr", "lstat", "lstat64", "madvise", "mbind", "memfd_create", "migrate_pages", "mincore", "mkdir", "mkdirat", "mknod", "mknodat", "mlock", "mlock2", "mlockall", "mmap", "mmap2", "modify_ldt", "mount", "move_pages", "mprotect", "mq_getsetattr", "mq_notify", "mq_open", "mq_timedreceive", "mq_timedsend", "mq_unlink", "mremap", "msgctl", "msgget", "msgrcv", "msgsnd", "msync", "munlock", "munlockall", "munmap", "name_to_handle_at", "nanosleep", "nfsservctl", "nice", "oldfstat", "oldlstat", "oldolduname", "oldstat", "olduname", "open", "open_by_handle_at", "openat", "pause", "pciconfig_iobase", "pciconfig_read", "pciconfig_write", "perf_event_open", "personality", "perfctr", "perfmonctl", "pipe", "pipe2", "pivot_root", "pkey_alloc", "pkey_free", "pkey_mprotect", "poll", "ppc_rtas", "ppc_swapcontext", "ppoll", "prctl", "pread64", "preadv", "preadv2", "prlimit64", "process_vm_readv", "process_vm_writev", "pselect6", "ptrace", "pwrite64", "pwritev", "pwritev2", "query_module", "quotactl", "read", "readahead", "readdir", "readlink", "readlinkat", "readv", "reboot", "recv", "recvfrom", "recvmsg", "recvmmsg", "remap_file_pages", "removexattr", "rename", "renameat", "renameat2", "request_key", "restart_syscall", "rmdir", "rt_sigaction", "rt_sigpending", "rt_sigprocmask", "rt_sigqueueinfo", "rt_sigreturn", "rt_sigsuspend", "rt_sigtimedwait", "rt_tgsigqueueinfo", "s390_runtime_instr", "s390_pci_mmio_read", "s390_pci_mmio_write", "sched_get_priority_max", "sched_get_priority_min", "sched_getaffinity", "sched_getattr", "sched_getparam", "sched_getscheduler", "sched_rr_get_interval", "sched_setaffinity", "sched_setattr", "sched_setparam", "sched_setscheduler", "sched_yield", "seccomp", "select", "semctl", "semget", "semop", "semtimedop", "send", "sendfile", "sendfile64", "sendmmsg", "sendmsg", "sendto", "set_mempolicy", "set_robust_list", "set_thread_area", "set_tid_address", "setdomainname", "setfsgid", "setfsgid32", "setfsuid", "setfsuid32", "setgid", "setgid32", "setgroups", "setgroups32", "sethostname", "setitimer", "setns", "setpgid", "setpriority", "setregid", "setregid32", "setresgid", "setresgid32", "setresuid", "setresuid32", "setreuid", "setreuid32", "setrlimit", "setsid", "setsockopt", "settimeofday", "setuid", "setuid32", "setup", "setxattr", "sgetmask", "shmat", "shmctl", "shmdt", "shmget", "shutdown", "sigaction", "sigaltstack", "signal", "signalfd", "signalfd4", "sigpending", "sigprocmask", "sigreturn", "sigsuspend", "socket", "socketcall", "socketpair", "splice", "spu_create", "spu_run", "ssetmask", "stat", "stat64", "statfs", "statfs64", "stime", "subpage_prot", "swapoff", "swapon", "symlink", "symlinkat", "sync", "sync_file_range", "sync_file_range2", "syncfs", "sysfs", "sysinfo", "syslog", "tee", "tgkill", "time", "timer_create", "timer_delete", "timer_getoverrun", "timer_gettime", "timer_settime", "timerfd_create", "timerfd_gettime", "timerfd_settime", "times", "tkill", "truncate", "truncate64", "ugetrlimit", "umask", "umount", "umount2", "uname", "unlink", "unlinkat", "unshare", "uselib", "ustat", "userfaultfd", "utime", "utimensat", "utimes", "utrap_install", "vfork", "vhangup", "vm86old", "vm86", "vmsplice", "wait4", "waitid", "waitpid", "write", "writev"]

def get_dict_sequence(trace,term_dict):
    dict_sequence = []
    for syscall in trace:
        if syscall in term_dict:
            dict_sequence.append(term_dict[syscall])
        else:
            dict_sequence.append(term_dict['unk'])
    return dict_sequence

def get_syscall_dict(ngrams_dict):
    syscall_dict = {}
    i = 0
    for ngram in ngrams_dict:
        if len(ngram.split()) == 1:
            syscall_dict[ngram] = i
            i+=1
    return syscall_dict

def create_vectorizers(corpus, base_dict_path):
    os.makedirs(base_dict_path, exist_ok=True)
    for i in range(1, max_n_gram):
        cvName = 'countvectorizer_ngram{}.pk'.format(i)
        tvName = 'tfidfvectorizer_ngram{}.pk'.format(i)
        hvName = 'hashingvectorizer_ngram{}.pk'.format(i)
        ndName = 'ngrams_dict_ngram{}.pk'.format(i)
        sdName = 'syscall_dict_ngram{}.pk'.format(i)


        countvectorizer = CountVectorizer(ngram_range=(i, i)).fit(corpus)
        pickle.dump(countvectorizer, open(base_dict_path + "\\" + cvName, "wb"))

        ngrams_dict = countvectorizer.vocabulary_
        pickle.dump(ngrams_dict, open(base_dict_path + "\\" + ndName, "wb"))

        tfidfvectorizer = TfidfVectorizer(ngram_range=(i, i), vocabulary=ngrams_dict).fit(corpus)
        pickle.dump(tfidfvectorizer, open(base_dict_path + "\\" + tvName, "wb"))

        if i == 1:
            syscall_dict = get_syscall_dict(ngrams_dict)
            pickle.dump(syscall_dict, open(base_dict_path + "\\" + sdName, "wb"))

            hashingvectorizer = HashingVectorizer(n_features=2**5).fit(corpus)  
            pickle.dump(hashingvectorizer, open(base_dict_path + "\\" + hvName, "wb"))
        
# Transform list of strings (system call instruction) to one long string of system calls
def from_trace_to_longstr(syscall_trace):
    tracestr = ''
    for syscall in syscall_trace:
        if syscall in syscall_list:
            tracestr += syscall + ' '
    # print(tracestr)
    return tracestr

# Either read raw data or the already created pickle files
def read_all_rawdata(rawdataPath, rawFileNames):
    corpus_dataframe, corpus = [],[]
    # Check if there exist any pickle files, read them if yes
    if any('.pk' in fileName for fileName in rawFileNames):
        print("reading data from pickle files")
        loc=open(rawdataPath + "\\" + 'corpus_dataframe.pk','rb')
        corpus_dataframe = pickle.load(loc)
        loc=open(rawdataPath + "\\" + 'corpus.pk','rb')
        corpus = pickle.load(loc)

    # If there are no pickle files yet, read the raw data
    else:
        print("reading raw data")
        # Create a bar to visualize the progress
        par = tqdm.tqdm(total=len(rawFileNames), ncols=100)

        # Go through the raw data files and create a list with the sequence of system call instructions
        for fn in rawFileNames:
            if '.csv' in fn:
                par.update(1)
                file = rawdataPath + "/" + fn
                trace = pd.read_csv(file)
                tr = trace.iloc[:,2].tolist()          
                longstr = from_trace_to_longstr(tr)
                corpus_dataframe.append(trace)
                corpus.append(longstr)

        # Create pickle files (so it's faster next time)
        pickle.dump(corpus, open(rawdataPath + "\\" + 'corpus.pk', "wb"))
        pickle.dump(corpus_dataframe, open(rawdataPath + "\\" + 'corpus_dataframe.pk', "wb"))
        par.close()
    return corpus_dataframe, corpus

def equalize_list_length(uneq_list):
    shortened_list = []
    #minLength = min(len(x) for x in uneq_list)
    minLength = 2500
    for feature in uneq_list:
        shortened_list.append(feature[:minLength])
    return shortened_list

def  create_onehot_dictionary(syscall_dict):
    onehot_dict = syscall_dict.copy()
    zero_list = [0] * len(syscall_dict.keys())
    for index, key in enumerate(onehot_dict):
        zero_list[index] = 1
        onehot_dict[key] = zero_list
        zero_list = [0] * len(syscall_dict.keys())
    return onehot_dict

def write_to_csv(encoded_trace_df, feature_path):
    if not os.path.exists(feature_path):
        os.makedirs(feature_path, exist_ok=True)
    for i in range(2, len(encoded_trace_df.columns)):
        print("writing", encoded_trace_df.columns[i])
        file_name = encoded_trace_df.columns[i] + ".csv"
        df = encoded_trace_df[["ids", "maltype"]]
        df_list = encoded_trace_df[encoded_trace_df.columns[i]].apply(lambda x: x.tolist())
        df.insert(2, encoded_trace_df.columns[i], df_list, True)
        df.to_csv(feature_path + "\\" + file_name, sep="\t", index=False)

def read_dicts(dictPath):
    vectorizers = {}
    dicts = {}
    for i in range(1, max_n_gram):
        cvName = 'countvectorizer_ngram{}'.format(i)
        tvName = 'tfidfvectorizer_ngram{}'.format(i)
        hvName = 'hashingvectorizer_ngram{}'.format(i)
        ndName = 'ngrams_dict_ngram{}'.format(i)
        sdName = 'syscall_dict_ngram{}'.format(i)

        loc=open(dictPath + "\\" + cvName+'.pk','rb')
        cv = pickle.load(loc)
        vectorizers[cvName] = cv

        loc=open(dictPath + "\\" + tvName+'.pk','rb')
        tv = pickle.load(loc)
        vectorizers[tvName] = tv

        loc=open(dictPath + "\\" + ndName+'.pk','rb')
        nd = pickle.load(loc)
        dicts[ndName] = nd

        if i == 1:
            loc=open(dictPath + "\\" + hvName+'.pk','rb')
            hv = pickle.load(loc)
            vectorizers[hvName] = hv

            loc=open(dictPath + "\\" + sdName+'.pk','rb')
            sd = pickle.load(loc)
            dicts[sdName] = sd

    return vectorizers, dicts

def get_features(argv):
    # Creating variable names for the corresponding folders and later used variables
    folder = argv[0]
    dirname = os.path.dirname(__file__)
    rawdataPath = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/raw/"))
    base_dict_path = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/features/pickle/"))
    feature_path = os.path.abspath(os.path.join(dirname, "../data/"+folder+"/features/csv/"))

    # if the encoded dataframe has been made before --> load it
    if False: #os.path.exists(base_dict_path + "\\" + "encoded_trace_df.pk"):
        print("Found encoded dataframe, loading it...")
        loc=open(base_dict_path + "\\" + "encoded_trace_df.pk",'rb')
        encoded_trace_df = pickle.load(loc)

        write_to_csv(encoded_trace_df, feature_path)

        return 0
        
    # if not --> create it
    else:
        rawFileNames = os.listdir(rawdataPath)
        features, columns = [], []
        ids, maltype = [], []

        # Create the initial dataframe with one column representing a sample and a column with it's corresponding behaviour
        for fileName in rawFileNames:
            if '.csv' in fileName:
                fis = fileName.split('_')
                fn = fis[0]
                i = '{}_{}_{}'.format(fis[0], fis[1], fis[2])
                maltype.append(fn)
                ids.append(i)
        features.append(ids)
        columns.append("ids")
        features.append(maltype)
        columns.append("maltype")

        # Read the data (either from the pickle files that have been created previously or the raw data)
        print('start to read rawdata')
        corpus_dataframe, corpus = read_all_rawdata(rawdataPath, rawFileNames)
        print('got rawdata')

        # Create Vectorizers and dictionaries if they don't exist yet
        if not os.path.exists(base_dict_path):
            print("creating vectorizers")
            create_vectorizers(corpus, base_dict_path)
        print("loading vectorizers")

        # Read in the vectorizers and dictionaries
        vectorizers, dicts = read_dicts(base_dict_path)
        print('get dicts finished!')

        # for key, value in vectorizers.items():
        #     print(key, ' : ', value)
        # for key, value in dicts.items():
        #     print(key, ' : ', value)

        syscall_dict = dicts["syscall_dict_ngram1"]
        onehot_dict = create_onehot_dictionary(syscall_dict)
        index_sequence_features = []
        onehot_sequence_features = []

        print(syscall_dict)

        for trace in corpus_dataframe:
            trace_list = trace.iloc[:,2].tolist()
            index_sequence_feature = [syscall_dict.get(item,item) for item in trace_list if item in syscall_list]
            onehot_sequence_feature = [onehot_dict.get(item,item) for item in trace_list if item in syscall_list]
            index_sequence_features.append(index_sequence_feature)
            onehot_sequence_features.append(onehot_sequence_feature)
      
        index_sequence_shortened = equalize_list_length(index_sequence_features)
        onehot_sequence_features_shortened = equalize_list_length(onehot_sequence_features)

        features.append(np.array(index_sequence_shortened))
        columns.append("index_sequence_features")
        features.append(np.array(onehot_sequence_features_shortened))
        columns.append("onehot_sequence_features")

        # use the vectorizers to create the actual features
        for i in range(1, max_n_gram):
            cvName = 'countvectorizer_ngram{}'.format(i)
            tvName = 'tfidfvectorizer_ngram{}'.format(i)

            cv = vectorizers[cvName]
            tv = vectorizers[tvName]

            # the frequency features
            frequency_features = cv.transform(corpus)
            frequency_features = frequency_features.toarray()

            # the tfidf features
            tfidf_features = tv.transform(corpus)
            tfidf_features = tfidf_features.toarray()

            features.append(frequency_features)
            columns.append(cvName)
            features.append(tfidf_features)
            columns.append(tvName)

        # hashed feature
        hvName = 'hashingvectorizer_ngram{}'.format(1)
        hv = vectorizers[hvName]
        hashing_features = hv.transform(corpus)
        hashing_features = hashing_features.toarray()
        features.append(hashing_features)     
        columns.append(hvName)
            
        encoded_trace_df = pd.DataFrame(features).transpose()
        encoded_trace_df.columns = columns
        print(encoded_trace_df)

        # store the created features in a pickle file
        pickle.dump(encoded_trace_df, open(base_dict_path + "\\" + 'encoded_trace_df.pk', "wb"))
        write_to_csv(encoded_trace_df, feature_path)

if __name__ == "__main__":
    get_features(sys.argv[1:])