#include <iostream>
#include <memory>
#include <vector>
#include <string>

// 演示RAII原则的资源管理类
class ResourceManager {
private:
    int* data;
    size_t size;

public:
    explicit ResourceManager(size_t n) : size(n) {
        std::cout << "分配内存: " << size * sizeof(int) << " 字节" << std::endl;
        data = new int[size];
    }

    ~ResourceManager() {
        std::cout << "释放内存: " << size * sizeof(int) << " 字节" << std::endl;
        delete[] data;
    }

    // 禁止拷贝构造和赋值操作，防止重复释放
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;

    void setValue(size_t index, int value) {
        if (index < size) {
            data[index] = value;
        }
    }

    int getValue(size_t index) const {
        return (index < size) ? data[index] : -1;
    }
};

// 使用智能指针管理的类
class MyClass {
public:
    MyClass(const std::string& name) : name_(name) {
        std::cout << "创建 MyClass 对象: " << name_ << std::endl;
    }

    ~MyClass() {
        std::cout << "销毁 MyClass 对象: " << name_ << std::endl;
    }

    void doSomething() {
        std::cout << name_ << " 正在执行操作" << std::endl;
    }

private:
    std::string name_;
};

// 演示内存泄漏的函数（不推荐的做法）
void memoryLeakDemo() {
    int* leakedMemory = new int[1000];  // 内存泄漏：没有对应的delete
    std::cout << "警告：这里发生了内存泄漏！" << std::endl;
}

// 演示智能指针的使用
void smartPointerDemo() {
    std::cout << "\n=== 智能指针演示 ===" << std::endl;

    // unique_ptr示例 - 独占所有权
    std::cout << "\n1. unique_ptr 示例:" << std::endl;
    {
        std::unique_ptr<MyClass> unique(new MyClass("Unique对象"));
        unique->doSomething();
        // unique_ptr离开作用域时自动释放内存
    }

    // shared_ptr示例 - 共享所有权
    std::cout << "\n2. shared_ptr 示例:" << std::endl;
    {
        std::shared_ptr<MyClass> shared1(new MyClass("Shared对象"));
        {
            std::shared_ptr<MyClass> shared2 = shared1;  // 增加引用计数
            std::cout << "当前引用计数: " << shared1.use_count() << std::endl;
            shared2->doSomething();
        }  // shared2离开作用域，引用计数减1
        std::cout << "当前引用计数: " << shared1.use_count() << std::endl;
    }  // shared1离开作用域，对象被销毁

    // weak_ptr示例 - 防止循环引用
    std::cout << "\n3. weak_ptr 示例:" << std::endl;
    {
        std::shared_ptr<MyClass> shared(new MyClass("Weak引用对象"));
        std::weak_ptr<MyClass> weak = shared;
        
        if (auto locked = weak.lock()) {
            locked->doSomething();
        }
        shared.reset();  // 释放shared_ptr
        if (weak.expired()) {
            std::cout << "weak_ptr 已过期" << std::endl;
        }
    }
}

int main() {
    std::cout << "=== 内存管理演示程序 ===\n" << std::endl;

    // 演示RAII原则
    std::cout << "1. RAII原则演示:" << std::endl;
    {
        ResourceManager rm(5);
        rm.setValue(0, 42);
        std::cout << "值: " << rm.getValue(0) << std::endl;
    }  // ResourceManager自动释放资源

    // 演示智能指针
    smartPointerDemo();

    // 演示内存泄漏（不推荐的做法）
    std::cout << "\n3. 内存泄漏演示（不推荐的做法）:" << std::endl;
    memoryLeakDemo();

    return 0;
}