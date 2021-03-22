#ifndef _TEST_H_
#define _TEST_H_

// 测试 extern inline 关键字组合的使用
extern inline void test();

// 将文本格式数据集转换成2进制数据集
void textToBinary();

// 测试从字符数组中解析数值
void testParseChars();

// 之前的众多测试
void tests(int argc, char* argv[]);

#endif // _TEST_H_