// mkerrors_nacl.sh /home/rsc/pub/nacl/native_client/src/trusted/service_runtime/include/sys/errno.h
// MACHINE GENERATED BY THE COMMAND ABOVE; DO NOT EDIT

package syscall

// TODO(brainman): populate errors in zerrors_windows.go

const (
	ERROR_FILE_NOT_FOUND      = 2
	ERROR_NO_MORE_FILES       = 18
	ERROR_BROKEN_PIPE         = 109
	ERROR_INSUFFICIENT_BUFFER = 122
	ERROR_MOD_NOT_FOUND       = 126
	ERROR_PROC_NOT_FOUND      = 127
	ERROR_DIRECTORY           = 267
	ERROR_IO_PENDING          = 997
	// TODO(brainman): should use value for EWINDOWS that does not clashes with anything else
	EWINDOWS = 99999 /* otherwise unused */
)

// TODO(brainman): fix all needed for os

const (
	EPERM           = 1
	ENOENT          = 2
	ESRCH           = 3
	EINTR           = 4
	EIO             = 5
	ENXIO           = 6
	E2BIG           = 7
	ENOEXEC         = 8
	EBADF           = 9
	ECHILD          = 10
	EAGAIN          = 11
	ENOMEM          = 12
	EACCES          = 13
	EFAULT          = 14
	EBUSY           = 16
	EEXIST          = 17
	EXDEV           = 18
	ENODEV          = 19
	ENOTDIR         = ERROR_DIRECTORY
	EISDIR          = 21
	EINVAL          = 22
	ENFILE          = 23
	EMFILE          = 24
	ENOTTY          = 25
	EFBIG           = 27
	ENOSPC          = 28
	ESPIPE          = 29
	EROFS           = 30
	EMLINK          = 31
	EPIPE           = 32
	ENAMETOOLONG    = 36
	ENOSYS          = 38
	EDQUOT          = 122
	EDOM            = 33
	ERANGE          = 34
	ENOMSG          = 35
	ECHRNG          = 37
	EL3HLT          = 39
	EL3RST          = 40
	ELNRNG          = 41
	EUNATCH         = 42
	ENOCSI          = 43
	EL2HLT          = 44
	EDEADLK         = 45
	ENOLCK          = 46
	EBADE           = 50
	EBADR           = 51
	EXFULL          = 52
	ENOANO          = 53
	EBADRQC         = 54
	EBADSLT         = 55
	EBFONT          = 57
	ENOSTR          = 60
	ENODATA         = 61
	ETIME           = 62
	ENOSR           = 63
	ENONET          = 64
	ENOPKG          = 65
	EREMOTE         = 66
	ENOLINK         = 67
	EADV            = 68
	ESRMNT          = 69
	ECOMM           = 70
	EPROTO          = 71
	EMULTIHOP       = 74
	ELBIN           = 75
	EDOTDOT         = 76
	EBADMSG         = 77
	EFTYPE          = 79
	ENOTUNIQ        = 80
	EBADFD          = 81
	EREMCHG         = 82
	ELIBACC         = 83
	ELIBBAD         = 84
	ELIBSCN         = 85
	ELIBMAX         = 86
	ELIBEXEC        = 87
	ENMFILE         = 89
	ENOTEMPTY       = 90
	ELOOP           = 92
	EOPNOTSUPP      = 95
	EPFNOSUPPORT    = 96
	ECONNRESET      = 104
	ENOBUFS         = 105
	EAFNOSUPPORT    = 106
	EPROTOTYPE      = 107
	ENOTSOCK        = 108
	ENOPROTOOPT     = 109
	ESHUTDOWN       = 110
	ECONNREFUSED    = 111
	EADDRINUSE      = 112
	ECONNABORTED    = 113
	ENETUNREACH     = 114
	ENETDOWN        = 115
	ETIMEDOUT       = 116
	EHOSTDOWN       = 117
	EHOSTUNREACH    = 118
	EINPROGRESS     = 119
	EALREADY        = 120
	EDESTADDRREQ    = 121
	EPROTONOSUPPORT = 123
	ESOCKTNOSUPPORT = 124
	EADDRNOTAVAIL   = 125
	ENETRESET       = 126
	EISCONN         = 127
	ENOTCONN        = 128
	ETOOMANYREFS    = 129
	EPROCLIM        = 130
	EUSERS          = 131
	EWOULDBLOCK     = 141
	ESTALE          = 133
	ENOMEDIUM       = 135
	ENOSHARE        = 136
	ECASECLASH      = 137
	EILSEQ          = 138
	EOVERFLOW       = 139
	ECANCELED       = 140
	EL2NSYNC        = 88
	EIDRM           = 91
	EMSGSIZE        = 132
)