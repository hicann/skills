# 代码审查技能文件 - 内存与指针安全

本文档例举C++安全编码规范中内存与指针安全相关条款, 为Ascend C 代码检视过程提供编码规范指导


# 二、C++安全编码规范 - 内存与指针安全

内存与指针安全涉及未初始化变量、悬空指针、数组越界、空指针解引用等关键安全问题


### 2.4 禁止使用未初始化的变量

【描述】
这里的变量，指的是局部动态变量，并且还包括内存堆上申请的内存块。 因为他们的初始值都是不可预料的，所以禁止未经有效初始化就直接读取其值。

```
void foo( ...)
{
	int data;
	bar(data); // 错误：未初始化就使用
	...
}
```

如果有不同分支，要确保所有分支都得到初始化后才能使用：

```
#define CUSTOMIZED_SIZE 100
void foo( ...)
{
	int data;
	if (condition > 0) {
		data = CUSTOMIZED_SIZE;
	}

	bar(data); // 错误：部分分支该值未初始化
	...
}
```

### 2.5 指向资源句柄或描述符的变量，在资源释放后立即赋予新值

**【描述】**
指向资源句柄或描述符的变量包括指针、文件描述符、socket描述符以及其它指向资源的变量。
以指针为例，当指针成功申请了一段内存之后，在这段内存释放以后，如果其指针未立即设置为NULL，也未分配一个新的对象，那这个指针就是一个悬空指针。
如果再对悬空指针操作，可能会发生重复释放或访问已释放内存的问题，造成安全漏洞。
消减该漏洞的有效方法是将释放后的指针立即设置为一个确定的新值，例如设置为NULL。对于全局性的资源句柄或描述符，在资源释放后，应该马上设置新值，以避免使用其已释放的无效值；对于只在单个函数内使用的资源句柄或描述符，应确保资源释放后其无效值不被再次使用。

**【错误代码示例】**
如下代码示例中，根据消息类型处理消息，处理完后释放掉body或head指向的内存，但是释放后未将指针设置为NULL。如果还有其他函数再次处理该消息结构体时，可能出现重复释放内存或访问已释放内存的问题。

```
int foo(void)
{
	SomeStruct *msg = NULL;
	... // 初始化msg->type，分配 msg->body 的内存空间
	if (msg->type == MESSAGE_A) {
		...
		free(msg->body);
	}

	...
EXIT:
	...
	free(msg->body);
	return ret;
}
```

**【正确代码示例】**
如下代码示例中，立即对释放后的指针设置为NULL，避免重复放指针。

```
int foo(void)
{
	SomeStruct  *msg = NULL;
	... // 初始化msg->type，分配 msg->body 的内存空间

	if (msg->type == MESSAGE_A) {
		...
		free(msg->body);
		msg->body = NULL;
	}

	...
EXIT:
	...
	free(msg->body);
	return ret;
}
```

当free()函数的入参为NULL时，函数不执行任何操作。

**【错误代码示例】**
如下代码示例中文件描述符关闭后未赋新值。

```
SOCKET s = INVALID_SOCKET;
int fd = -1;
...
closesocket(s);
...
close(fd);
...
```

**【正确代码示例】**
如下代码示例中，在资源释放后，对应的变量应该立即赋予新值。

```
SOCKET s = INVALID_SOCKET;
int fd = -1;
...
closesocket(s);
s = INVALID_SOCKET;
...
close(fd);
fd = -1;
...
```

### 2.6 外部数据作为数组索引时必须确保在数组大小范围内

**【描述】**
外部数据作为数组索引对内存进行访问时，必须对数据的大小进行严格的校验，确保数组数组索引在有效范围内，否则会导致严重的错误。
当一个指针指向数组元素时，可以指向数组最后一个元素的下一个元素的位置，但是不能读写该位置的内存。

**【错误代码示例】**
如下代码示例中, set_dev_id()函数存在差一错误，当 index 等于 DEV_NUM 时，恰好越界写一个元素；
同样get_dev()函数也存在差一错误，虽然函数执行过程中没有问题，但是当解引用这个函数返回的指针时，行为是未定义的。

```
#define DEV_NUM 10
#define MAX_NAME_LEN 128
typedef struct {
	int id;
	char name[MAX_NAME_LEN];
} Dev;

static Dev devs[DEV_NUM];
int set_dev_id(size_t index, int id)
{
	if (index > DEV_NUM) { // 错误：差一错误。
 		... // 错误处理
	}

	devs[index].id = id;
	return 0;
}

static Dev *get_dev(size_t index)
{
	if (index > DEV_NUM) { // 错误：差一错误。
 		... // 错误处理
	}

	return devs + index;
}
```

**【正确代码示例】**
如下代码示例中，修改校验索引的条件，避免差一错误。

```
#define DEV_NUM 10
#define MAX_NAME_LEN 128
typedef struct {
	int id;
	char name[MAX_NAME_LEN];
} Dev;

static Dev devs[DEV_NUM];

int set_dev_Id (size_t index, int id)
{
	if (index >= DEV_NUM) {
		... // 错误处理
	}

	devs[index].id = id;
	return 0;
}

static Dev *get_dev(size_t index)
{
	if (index >= DEV_NUM) {
 		... // 错误处理
	}

	return devs + index;
}
```

**【相关软件CWE编号】** CWE-119，CWE-123，CWE-125

### 2.7 禁止通过对指针变量进行sizeof操作来获取数组大小

**【描述】**
将指针当做数组进行sizeof操作时，会导致实际的执行结果与预期不符。例如：变量定义 char *p = array，其中array的定义为char array[LEN]，表达式sizeof(p) 得到的结果与 sizeof(char *)相同，并非array的长度。

**【错误代码示例】**
如下代码示例中，buffer和path分别是指针和数组，程序员想对这2个内存进行清0操作，但由于程序员的疏忽，将内存大小了误写成了sizeof(buffer)，与预期不符。

```
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
// sizeof与预期不符，其结果为指针本身的大小而不是缓冲区大小
(void)memset(buffer, 0, sizeof(buffer));
```

**【正确代码示例】**
如下代码示例中，将sizeof(buffer)修改为申请的缓冲区大小：

```
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
(void)memset(buffer, 0, SIZE); // 使用申请的缓冲区大小
```

### 2.8 指针操作，使用前必须要判空

**【描述】**解引用空指针会导致程序产生未定义行为，通常会造成程序异常终止。

* 指针变量在使用前，一定要做好初始化的赋值，严禁对空指针进行访问；
* 对于指针所代表的地址空间的任何操作，一定要保证空间的有效性；
* 指针指向的内存释放后，操作系统并不会自动将指针置为NULL，需要调用者将指针显式置为NULL，防止"野指针"
* 内部函数传参时，在上级调用函数能确保传参不会为NULL的情况下，可以不对入参进行非NULL检查。
  **【错误代码示例】**

```
// result 未做除空指针判断
auto result = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
INFER_SHAPE(BinaryCrossEntropy, OP_INPUT(self, target, weight), OP_OUTPUT(result), OP_ATTR(reduction));
```

**【正确代码示例】**

```
// 业务中较多场景会需要申请指针，可以在申请加校验，保证业务场景不会触发空指针的解引用
auto result = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
CHECK_RET(reluOut != nullptr, nullptr);
INFER_SHAPE(BinaryCrossEntropy, OP_INPUT(self, target, weight), OP_OUTPUT(result), OP_ATTR(reduction));
```

**【错误代码示例】**

```
// 未对context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX)进行非空判断，导致coredump
uint32_t attenMaskTypeSize = NUM_BYTES_BOOL; // default DT_BOOL
auto attenMaskInput = context.GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);
if (attenMaskInput != nullptr) {
   auto attenMaskDataType = context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX)->GetDataType();
   ......
}
```

**【正确代码示例】**

```
uint32_t attenMaskTypeSize = NUM_BYTES_BOOL; // default DT_BOOL
auto attenMaskInput = context.GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);
if (attenMaskInput != nullptr) {
    OP_TILING_CHECK(context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX) == nullptr,
                VECTOR_INNER_ERR_REPORT_TILIING(context.GetNodeName(), "Desc of tensor atten_mask is nullptr"),
                return ge::GRAPH_FAILED);  // 校验空指针，报错退出

    auto attenMaskDataType = context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX)->GetDataType();
    ......}
```

**【相关软件CWE编号】** CWE-170，CWE-464
