#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>

// ================ 创建型模式 ================

// 1. 单例模式
class Singleton {
private:
    static Singleton* instance;
    Singleton() = default;

public:
    static Singleton* getInstance() {
        if (instance == nullptr) {
            instance = new Singleton();
        }
        return instance;
    }

    void showMessage() {
        std::cout << "单例模式：我是唯一的实例" << std::endl;
    }
};

Singleton* Singleton::instance = nullptr;

// 2. 工厂方法模式
class Product {
public:
    virtual void use() = 0;
    virtual ~Product() = default;
};

class ConcreteProductA : public Product {
public:
    void use() override {
        std::cout << "使用产品A" << std::endl;
    }
};

class ConcreteProductB : public Product {
public:
    void use() override {
        std::cout << "使用产品B" << std::endl;
    }
};

class Factory {
public:
    virtual Product* createProduct() = 0;
    virtual ~Factory() = default;
};

class ConcreteFactoryA : public Factory {
public:
    Product* createProduct() override {
        return new ConcreteProductA();
    }
};

class ConcreteFactoryB : public Factory {
public:
    Product* createProduct() override {
        return new ConcreteProductB();
    }
};

// ================ 结构型模式 ================

// 3. 适配器模式
class Target {
public:
    virtual void request() = 0;
    virtual ~Target() = default;
};

class Adaptee {
public:
    void specificRequest() {
        std::cout << "适配器模式：特殊请求" << std::endl;
    }
};

class Adapter : public Target {
private:
    std::unique_ptr<Adaptee> adaptee;

public:
    Adapter() : adaptee(std::make_unique<Adaptee>()) {}

    void request() override {
        std::cout << "适配器模式：转换请求 -> ";
        adaptee->specificRequest();
    }
};

// 4. 装饰器模式
class Component {
public:
    virtual void operation() = 0;
    virtual ~Component() = default;
};

class ConcreteComponent : public Component {
public:
    void operation() override {
        std::cout << "装饰器模式：基础组件" << std::endl;
    }
};

class Decorator : public Component {
protected:
    std::unique_ptr<Component> component;

public:
    explicit Decorator(Component* c) : component(c) {}

    void operation() override {
        if (component) {
            component->operation();
        }
    }
};

class ConcreteDecorator : public Decorator {
public:
    explicit ConcreteDecorator(Component* c) : Decorator(c) {}

    void operation() override {
        std::cout << "装饰器模式：添加额外功能 -> ";
        Decorator::operation();
    }
};

// ================ 行为型模式 ================

// 5. 观察者模式
class Observer {
public:
    virtual void update(const std::string& message) = 0;
    virtual ~Observer() = default;
};

class Subject {
private:
    std::vector<Observer*> observers;

public:
    void attach(Observer* observer) {
        observers.push_back(observer);
    }

    void detach(Observer* observer) {
        observers.erase(
            std::remove(observers.begin(), observers.end(), observer),
            observers.end());
    }

    void notify(const std::string& message) {
        for (auto observer : observers) {
            observer->update(message);
        }
    }
};

class ConcreteObserver : public Observer {
private:
    std::string name;

public:
    explicit ConcreteObserver(const std::string& n) : name(n) {}

    void update(const std::string& message) override {
        std::cout << "观察者" << name << "收到消息: " << message << std::endl;
    }
};

// 6. 策略模式
class Strategy {
public:
    virtual void execute() = 0;
    virtual ~Strategy() = default;
};

class ConcreteStrategyA : public Strategy {
public:
    void execute() override {
        std::cout << "策略模式：执行策略A" << std::endl;
    }
};

class ConcreteStrategyB : public Strategy {
public:
    void execute() override {
        std::cout << "策略模式：执行策略B" << std::endl;
    }
};

class Context {
private:
    std::unique_ptr<Strategy> strategy;

public:
    void setStrategy(Strategy* s) {
        strategy.reset(s);
    }

    void executeStrategy() {
        if (strategy) {
            strategy->execute();
        }
    }
};

// 主函数：演示各种设计模式
int main() {
    std::cout << "=== C++设计模式演示 ===\n" << std::endl;

    // 1. 单例模式演示
    std::cout << "1. 单例模式演示:" << std::endl;
    Singleton::getInstance()->showMessage();
    std::cout << std::endl;

    // 2. 工厂方法模式演示
    std::cout << "2. 工厂方法模式演示:" << std::endl;
    ConcreteFactoryA factoryA;
    ConcreteFactoryB factoryB;
    std::unique_ptr<Product> productA(factoryA.createProduct());
    std::unique_ptr<Product> productB(factoryB.createProduct());
    productA->use();
    productB->use();
    std::cout << std::endl;

    // 3. 适配器模式演示
    std::cout << "3. 适配器模式演示:" << std::endl;
    std::unique_ptr<Target> adapter = std::make_unique<Adapter>();
    adapter->request();
    std::cout << std::endl;

    // 4. 装饰器模式演示
    std::cout << "4. 装饰器模式演示:" << std::endl;
    Component* component = new ConcreteComponent();
    std::unique_ptr<Component> decorator(
        new ConcreteDecorator(component));
    decorator->operation();
    std::cout << std::endl;

    // 5. 观察者模式演示
    std::cout << "5. 观察者模式演示:" << std::endl;
    Subject subject;
    ConcreteObserver observer1("1号");
    ConcreteObserver observer2("2号");
    subject.attach(&observer1);
    subject.attach(&observer2);
    subject.notify("重要通知！");
    std::cout << std::endl;

    // 6. 策略模式演示
    std::cout << "6. 策略模式演示:" << std::endl;
    Context context;
    context.setStrategy(new ConcreteStrategyA());
    context.executeStrategy();
    context.setStrategy(new ConcreteStrategyB());
    context.executeStrategy();

    return 0;
}