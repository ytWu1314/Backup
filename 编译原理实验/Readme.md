# 实验一: C++ 单词拼装器

一、必做内容：

  1. 把C++源代码中的各类单词（记号）进行拼装分类。
    C++语言包含了几种类型的单词（记号）：标识符，关键字，数（包括整数、浮点数），字符串、注释、特殊符号（分界符）和运算符号	等 
    2. 要求应用程序应为Windows界面。
    3. 打开一个C++源文件，列出所有可以拼装的单词（记号）。
    4. 应该书写完善的软件设计文档。

# 实验二：XLEX生成器

一、实验内容：
设计一个应用软件，以实现将正则表达式-->NFA--->DFA-->DFA最小化-->词法分析程序

二、必做实验要求：
 （1）正则表达式应该支持单个字符，运算符号有： 连接  选择 闭包  括号
 （2）要提供一个源程序编辑界面，让用户输入正则表达式（可保存、打开源程序）
 （3）需要提供窗口以便用户可以查看转换得到的NFA（用状态转换表呈现即可）
 （4）需要提供窗口以便用户可以查看转换得到的DFA（用状态转换表呈现即可）
 （5）需要提供窗口以便用户可以查看转换得到的最小化DFA（用状态转换表呈现即可）
 （6）需要提供窗口以便用户可以查看转换得到的词法分析程序（该分析程序需要用C语言描述）
 （7）应该书写完善的软件文档

# 实验三：文法问题处理器

一、实验内容：
设计一个应用软件，以实现文法的化简及各种问题的处理。

二、实验要求：
 （1）系统需要提供一个文法编辑界面，让用户输入文法规则（可保存、打开存有文法规则的文件）
 （2）化简文法：检查文法是否存在有害规则和多余规则并将其去除。系统应该提供窗口以便用户可以查看文法化简后的结果。
 （3）检查该文法是否存在着左公共因子（可能包含直接和间接的情况）。如果存在，则消除该文法的左公共因子。系统应该提供窗口以便用户可以查看消除左公共因子的结果。
 （4）检查该文法是否存在着左递归（可能包含直接和间接的情况），如果存在，则消除该文法的左递归。系统应该提供窗口以便用户可以查看消除左递归后的结果。
 （5）求出经过前面步骤处理好的文法各非终结符号的first集合与follow集合，并提供窗口以便用户可以查看这些集合结果。【可以采用表格的形式呈现】
 （6）对输入的句子进行最左推导分析，系统应该提供界面让用户可以输入要分析的句子以及方便用户查看最左推导的每一步推导结果。【可以采用表格的形式呈现推导的每一步结果】
 （7）应该书写完善的软件文档

# 实验四：TINY扩充语言的语法分析

一、实验内容：

扩充的语法规则有：实现 do while循环，for循环，扩充算术表达式的运算符号：-= 减法赋值运算符号（类似于C语言的-=）、求余%、乘方^，
扩充比较运算符号：==（等于），>(大于)、<=(小于等于)、>=(大于等于)、<>(不等于)等运算符号，
新增支持正则表达式以及用于repeat循环、do while循环、if条件语句作条件判断的逻辑表达式：运算符号有 and（与）、 or（或）、 not（非） 。
具体文法规则自行构造。

可参考：云盘中参考书P97及P136的文法规则。

(1) Dowhile-stmt-->do  stmt-sequence  while(exp); 
(2) for-stmt-->for identifier:=simple-exp  to  simple-exp  do  stmt-sequence enddo    步长递增1
(3) for-stmt-->for identifier:=simple-exp  downto  simple-exp  do  stmt-sequence enddo    步长递减1
(4) -= 减法赋值运算符号、求余%、乘方^、>=(大于等于)、<=(小于等于)、>(大于)、<>(不等于)运算符号的文法规则请自行组织。
(5)把tiny原来的赋值运算符号(:=)改为(=),而等于的比较符号符号（=）则改为（==）
(6)为tiny语言增加一种新的表达式——正则表达式，其支持的运算符号有  或(|)  、连接(&)、闭包(#)、括号( ) 以及基本正则表达式 。
(7)为tiny语言增加一种新的语句，ID:=正则表达式  
(8)为tiny语言增加一种新的表达式——逻辑表达式，其支持的运算符号有  and(与)  、or (或)、非(not)。
(9)为了实现以上的扩充或改写功能，还需要对原tiny语言的文法规则做好相应的改造处理。