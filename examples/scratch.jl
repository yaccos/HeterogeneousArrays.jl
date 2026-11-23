struct MyWrapper1 <: Function
    f::Function
end

struct MyWrapper2{F<:Function} <: Function
    f::F
end

struct MyWrapper3{F<:Function} <: Function
    function MyWrapper3(f::Function)
        new{typeof(f)}()
    end
end

(::MyWrapper3{F})(args...; kwargs...) where {F} = F.instance(args...; kwargs...)

a = 10
f = x -> x + a

w = MyWrapper3(f)

w(32)

struct Foo{f} 
    function Foo(f)
        new{f}()
    end
end

(::Foo{f})(args...; kwargs...) where {f} = f(args...; kwargs...)

function return_f(a)
    a_vec = fill(a, 10)
    x -> x .+ a_vec
end

f = return_f(42)
F = typeof(f)
F.instance
Foo(f)

abstract type AbstractStorage end


mutable struct Stack{STACK_WIDTH_T <: Unsigned} <: AbstractStorage
    stack::Base.Memory{STACK_WIDTH_T}
    function Stack(STACK_WIDTH_T, stack_length, default_stack_value=zero(STACK_WIDTH_T))
        new{STACK_WIDTH_T}(fill(default_stack_value, stack_length))
    end
end
