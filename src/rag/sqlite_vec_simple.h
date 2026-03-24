#pragma once
#include <sqlite3.h>

// sqlite-vec 需要的函数声明
#ifdef __cplusplus
extern "C" {
#endif

	// 扩展初始化函数
	int sqlite3_vec_init(sqlite3* db, char** pzErrMsg, const sqlite3_api_routines* pApi);

#ifdef __cplusplus
}
#endif