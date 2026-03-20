# 代码审查技能文件 - 资源管理

本文档例举C++安全编码规范中资源管理相关条款, 为Ascend C 代码检视过程提供编码规范指导


# 二、C++安全编码规范 - 资源管理

资源管理涉及资源申请失败检查、内存/句柄/锁泄漏等关键安全问题


### 2.9 资源申请后必须判断是否成功

**【描述】**
内存、对象、stream、notify等资源申请分配一旦失败，那么后续的操作会存在未定义的行为风险。比如malloc申请失败返回了空指针，对空指针的解引用是一种未定义行为，需要进行异常捕获或返回值校验。

**【错误代码示例】**
如下代码示例中，调用malloc分配内存之后，没有判断是否成功，直接引用了p。如果malloc失败，它将返回一个空指针并传递给p。当如下代码在内存拷贝中解引用了该空指针p时，程序会出现未定义行为。

```
struct tm *make_tm(int year, int mon, int day, int hour, int min, int sec)
{
	struct tm *tmb = (struct tm*)malloc(sizeof(*tmb));
	tmb->year = year;
	...

	return tmb;
}
```

**【正确代码示例】**
如下代码示例中，在malloc分配内存之后，立即判断其是否成功，消除了上述的风险。

```
struct tm  *make_tm(int(int year, int mon, int day, int hour, int min, int sec)
{
	struct tm  *tmb = (struct tm *)malloc(sizeof(*tmb));
	if (tmb == NULL) {
		... // 错误处理
	}

	tmb->year = year;
	...
	return tmb;
}
```

### 2.12 资源泄露（内存、句柄、锁等）

**【描述】**

* 1、资源申请和释放必须匹配，包括：内存类的malloc/free/alloc_page/free_page, 锁的lock/unlock、文件的open/close，文件引用的fget/fput，symbol的symbol_get/symbol_put，内存的mm_get/mm_put，尤其关注异常分支和异常场景。
* 2、释放结构体/类/数组/各类数据容器指针前，必须先释放成员指针
* 3、对外接口处理涉及资源申请但未释放，引起资源泄露，导致拒绝服务
* 4、C++捕获异常时确保恢复程序的一致性; 建议使用RAII模式，确保资源在异常发生时自动释放
