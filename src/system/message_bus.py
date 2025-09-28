"""
📡 消息总线系统
生产级组件间通信系统，实现Redis消息队列、事件驱动架构、发布订阅等完整功能
支持高性能消息传递、事件路由、消息持久化和故障恢复
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from loguru import logger
from src.core.config import settings


class MessageType(Enum):
    """消息类型"""
    EVENT = "event"                 # 事件消息
    COMMAND = "command"             # 命令消息
    QUERY = "query"                 # 查询消息
    RESPONSE = "response"           # 响应消息
    NOTIFICATION = "notification"   # 通知消息
    HEARTBEAT = "heartbeat"         # 心跳消息


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Message:
    """消息对象"""
    id: str                                 # 消息ID
    type: MessageType                       # 消息类型
    topic: str                              # 主题
    payload: Any                            # 消息内容
    sender: str                             # 发送者
    recipient: Optional[str] = None         # 接收者
    priority: MessagePriority = MessagePriority.NORMAL  # 优先级
    timestamp: float = field(default_factory=time.time)  # 时间戳
    correlation_id: Optional[str] = None    # 关联ID
    reply_to: Optional[str] = None          # 回复地址
    ttl: Optional[int] = None               # 生存时间(秒)
    retry_count: int = 0                    # 重试次数
    max_retries: int = 3                    # 最大重试次数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass
class Subscription:
    """订阅信息"""
    subscriber_id: str                      # 订阅者ID
    topic: str                              # 主题
    handler: Callable[[Message], Any]      # 处理函数
    filter_func: Optional[Callable[[Message], bool]] = None  # 过滤函数
    created_at: float = field(default_factory=time.time)  # 创建时间
    message_count: int = 0                  # 处理消息数
    last_message_time: Optional[float] = None  # 最后消息时间


class MessageSerializer:
    """消息序列化器"""
    
    @staticmethod
    def serialize(message: Message) -> bytes:
        """序列化消息"""
        try:
            message_dict = {
                'id': message.id,
                'type': message.type.value,
                'topic': message.topic,
                'payload': message.payload,
                'sender': message.sender,
                'recipient': message.recipient,
                'priority': message.priority.value,
                'timestamp': message.timestamp,
                'correlation_id': message.correlation_id,
                'reply_to': message.reply_to,
                'ttl': message.ttl,
                'retry_count': message.retry_count,
                'max_retries': message.max_retries,
                'metadata': message.metadata
            }
            
            return pickle.dumps(message_dict)
            
        except Exception as e:
            logger.error(f"消息序列化失败: {e}")
            raise
    
    @staticmethod
    def deserialize(data: bytes) -> Message:
        """反序列化消息"""
        try:
            message_dict = pickle.loads(data)
            
            return Message(
                id=message_dict['id'],
                type=MessageType(message_dict['type']),
                topic=message_dict['topic'],
                payload=message_dict['payload'],
                sender=message_dict['sender'],
                recipient=message_dict.get('recipient'),
                priority=MessagePriority(message_dict['priority']),
                timestamp=message_dict['timestamp'],
                correlation_id=message_dict.get('correlation_id'),
                reply_to=message_dict.get('reply_to'),
                ttl=message_dict.get('ttl'),
                retry_count=message_dict.get('retry_count', 0),
                max_retries=message_dict.get('max_retries', 3),
                metadata=message_dict.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"消息反序列化失败: {e}")
            raise


class InMemoryMessageBus:
    """内存消息总线"""
    
    def __init__(self):
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.message_queue: deque = deque()
        self.dead_letter_queue: deque = deque(maxlen=1000)
        self.message_history: deque = deque(maxlen=10000)
        
        # 统计信息
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'active_subscriptions': 0
        }
        
        # 运行状态
        self.running = False
        self.worker_thread = None
        
        logger.info("内存消息总线初始化完成")
    
    def start(self):
        """启动消息总线"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._message_worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("内存消息总线已启动")
    
    def stop(self):
        """停止消息总线"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        
        logger.info("内存消息总线已停止")
    
    def _message_worker(self):
        """消息处理工作线程"""
        while self.running:
            try:
                if self.message_queue:
                    message = self.message_queue.popleft()
                    self._deliver_message(message)
                else:
                    time.sleep(0.001)  # 1ms休眠
                    
            except Exception as e:
                logger.error(f"消息处理失败: {e}")
                time.sleep(0.1)
    
    def publish(self, message: Message) -> bool:
        """发布消息"""
        try:
            # 检查TTL
            if message.ttl and (time.time() - message.timestamp) > message.ttl:
                logger.warning(f"消息已过期: {message.id}")
                return False
            
            # 添加到队列
            if message.priority == MessagePriority.CRITICAL:
                self.message_queue.appendleft(message)
            else:
                self.message_queue.append(message)
            
            self.stats['messages_sent'] += 1
            self.message_history.append(message)
            
            return True
            
        except Exception as e:
            logger.error(f"发布消息失败: {e}")
            return False
    
    def subscribe(self, topic: str, handler: Callable[[Message], Any], 
                  subscriber_id: str, filter_func: Optional[Callable[[Message], bool]] = None) -> bool:
        """订阅主题"""
        try:
            subscription = Subscription(
                subscriber_id=subscriber_id,
                topic=topic,
                handler=handler,
                filter_func=filter_func
            )
            
            self.subscriptions[topic].append(subscription)
            self.stats['active_subscriptions'] += 1
            
            logger.info(f"订阅成功: {subscriber_id} -> {topic}")
            return True
            
        except Exception as e:
            logger.error(f"订阅失败: {e}")
            return False
    
    def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """取消订阅"""
        try:
            if topic in self.subscriptions:
                self.subscriptions[topic] = [
                    sub for sub in self.subscriptions[topic] 
                    if sub.subscriber_id != subscriber_id
                ]
                
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
                
                self.stats['active_subscriptions'] -= 1
                logger.info(f"取消订阅: {subscriber_id} -> {topic}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"取消订阅失败: {e}")
            return False
    
    def _deliver_message(self, message: Message):
        """投递消息"""
        try:
            delivered = False
            
            # 查找订阅者
            for subscription in self.subscriptions.get(message.topic, []):
                try:
                    # 应用过滤器
                    if subscription.filter_func and not subscription.filter_func(message):
                        continue
                    
                    # 检查接收者
                    if message.recipient and message.recipient != subscription.subscriber_id:
                        continue
                    
                    # 调用处理函数
                    subscription.handler(message)
                    subscription.message_count += 1
                    subscription.last_message_time = time.time()
                    
                    delivered = True
                    
                except Exception as e:
                    logger.error(f"消息处理失败: {subscription.subscriber_id} - {e}")
            
            if delivered:
                self.stats['messages_delivered'] += 1
            else:
                # 重试逻辑
                if message.retry_count < message.max_retries:
                    message.retry_count += 1
                    self.message_queue.append(message)
                else:
                    # 移到死信队列
                    self.dead_letter_queue.append(message)
                    self.stats['messages_failed'] += 1
                    logger.warning(f"消息投递失败，移入死信队列: {message.id}")
            
        except Exception as e:
            logger.error(f"消息投递失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'stats': self.stats.copy(),
            'queue_size': len(self.message_queue),
            'dead_letter_size': len(self.dead_letter_queue),
            'history_size': len(self.message_history),
            'topics': list(self.subscriptions.keys()),
            'subscription_count': sum(len(subs) for subs in self.subscriptions.values())
        }


class RedisMessageBus:
    """Redis消息总线"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.async_redis_client = None
        self.pubsub = None
        
        # 订阅管理
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.subscriber_tasks: Dict[str, asyncio.Task] = {}
        
        # 统计信息
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'active_subscriptions': 0
        }
        
        # 运行状态
        self.running = False
        
        logger.info(f"Redis消息总线初始化: {redis_url}")
    
    async def start(self):
        """启动Redis消息总线"""
        try:
            if not REDIS_AVAILABLE:
                raise ImportError("Redis库未安装")
            
            # 创建Redis连接
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            self.async_redis_client = aioredis.from_url(self.redis_url, decode_responses=False)
            
            # 测试连接
            await self.async_redis_client.ping()
            
            self.running = True
            logger.info("Redis消息总线启动成功")
            
        except Exception as e:
            logger.error(f"Redis消息总线启动失败: {e}")
            raise
    
    async def stop(self):
        """停止Redis消息总线"""
        try:
            self.running = False
            
            # 停止所有订阅任务
            for task in self.subscriber_tasks.values():
                task.cancel()
            
            # 等待任务完成
            if self.subscriber_tasks:
                await asyncio.gather(*self.subscriber_tasks.values(), return_exceptions=True)
            
            # 关闭连接
            if self.async_redis_client:
                await self.async_redis_client.close()
            
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("Redis消息总线已停止")
            
        except Exception as e:
            logger.error(f"停止Redis消息总线失败: {e}")
    
    async def publish(self, message: Message) -> bool:
        """发布消息到Redis"""
        try:
            if not self.running:
                return False
            
            # 序列化消息
            serialized_message = MessageSerializer.serialize(message)
            
            # 发布到Redis
            channel = f"topic:{message.topic}"
            await self.async_redis_client.publish(channel, serialized_message)
            
            # 如果有TTL，设置过期时间
            if message.ttl:
                key = f"message:{message.id}"
                await self.async_redis_client.setex(key, message.ttl, serialized_message)
            
            self.stats['messages_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"发布消息到Redis失败: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable[[Message], Any], 
                       subscriber_id: str, filter_func: Optional[Callable[[Message], bool]] = None) -> bool:
        """订阅Redis主题"""
        try:
            subscription = Subscription(
                subscriber_id=subscriber_id,
                topic=topic,
                handler=handler,
                filter_func=filter_func
            )
            
            self.subscriptions[topic].append(subscription)
            
            # 创建订阅任务
            if topic not in self.subscriber_tasks:
                task = asyncio.create_task(self._subscribe_worker(topic))
                self.subscriber_tasks[topic] = task
            
            self.stats['active_subscriptions'] += 1
            logger.info(f"Redis订阅成功: {subscriber_id} -> {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Redis订阅失败: {e}")
            return False
    
    async def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """取消Redis订阅"""
        try:
            if topic in self.subscriptions:
                self.subscriptions[topic] = [
                    sub for sub in self.subscriptions[topic] 
                    if sub.subscriber_id != subscriber_id
                ]
                
                # 如果没有订阅者了，停止订阅任务
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
                    
                    if topic in self.subscriber_tasks:
                        self.subscriber_tasks[topic].cancel()
                        del self.subscriber_tasks[topic]
                
                self.stats['active_subscriptions'] -= 1
                logger.info(f"取消Redis订阅: {subscriber_id} -> {topic}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"取消Redis订阅失败: {e}")
            return False
    
    async def _subscribe_worker(self, topic: str):
        """Redis订阅工作任务"""
        try:
            channel = f"topic:{topic}"
            pubsub = self.async_redis_client.pubsub()
            await pubsub.subscribe(channel)
            
            logger.info(f"开始监听Redis频道: {channel}")
            
            async for redis_message in pubsub.listen():
                if not self.running:
                    break
                
                if redis_message['type'] == 'message':
                    try:
                        # 反序列化消息
                        message = MessageSerializer.deserialize(redis_message['data'])
                        
                        # 处理消息
                        await self._handle_redis_message(topic, message)
                        
                    except Exception as e:
                        logger.error(f"处理Redis消息失败: {e}")
            
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            
        except asyncio.CancelledError:
            logger.info(f"Redis订阅任务被取消: {topic}")
        except Exception as e:
            logger.error(f"Redis订阅工作任务失败: {topic} - {e}")
    
    async def _handle_redis_message(self, topic: str, message: Message):
        """处理Redis消息"""
        try:
            delivered = False
            
            for subscription in self.subscriptions.get(topic, []):
                try:
                    # 应用过滤器
                    if subscription.filter_func and not subscription.filter_func(message):
                        continue
                    
                    # 检查接收者
                    if message.recipient and message.recipient != subscription.subscriber_id:
                        continue
                    
                    # 调用处理函数
                    if asyncio.iscoroutinefunction(subscription.handler):
                        await subscription.handler(message)
                    else:
                        subscription.handler(message)
                    
                    subscription.message_count += 1
                    subscription.last_message_time = time.time()
                    
                    delivered = True
                    
                except Exception as e:
                    logger.error(f"Redis消息处理失败: {subscription.subscriber_id} - {e}")
            
            if delivered:
                self.stats['messages_delivered'] += 1
            else:
                self.stats['messages_failed'] += 1
            
        except Exception as e:
            logger.error(f"处理Redis消息失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'stats': self.stats.copy(),
            'topics': list(self.subscriptions.keys()),
            'subscription_count': sum(len(subs) for subs in self.subscriptions.values()),
            'active_tasks': len(self.subscriber_tasks)
        }


class MessageBus:
    """统一消息总线接口"""
    
    def __init__(self, use_redis: bool = True, redis_url: str = "redis://localhost:6379/0"):
        self.use_redis = use_redis and REDIS_AVAILABLE
        
        if self.use_redis:
            self.backend = RedisMessageBus(redis_url)
            logger.info("使用Redis消息总线")
        else:
            self.backend = InMemoryMessageBus()
            logger.info("使用内存消息总线")
        
        # 消息路由
        self.message_routes: Dict[str, str] = {}
        
        # 中间件
        self.middleware: List[Callable[[Message], Message]] = []
        
        logger.info("消息总线初始化完成")
    
    async def start(self):
        """启动消息总线"""
        if hasattr(self.backend, 'start'):
            if asyncio.iscoroutinefunction(self.backend.start):
                await self.backend.start()
            else:
                self.backend.start()
    
    async def stop(self):
        """停止消息总线"""
        if hasattr(self.backend, 'stop'):
            if asyncio.iscoroutinefunction(self.backend.stop):
                await self.backend.stop()
            else:
                self.backend.stop()
    
    def add_middleware(self, middleware: Callable[[Message], Message]):
        """添加中间件"""
        self.middleware.append(middleware)
    
    def add_route(self, pattern: str, target_topic: str):
        """添加消息路由"""
        self.message_routes[pattern] = target_topic
    
    def _apply_middleware(self, message: Message) -> Message:
        """应用中间件"""
        for middleware in self.middleware:
            message = middleware(message)
        return message
    
    def _route_message(self, message: Message) -> Message:
        """路由消息"""
        for pattern, target_topic in self.message_routes.items():
            if pattern in message.topic:
                message.topic = target_topic
                break
        return message
    
    async def publish(self, topic: str, payload: Any, sender: str, 
                     message_type: MessageType = MessageType.EVENT,
                     priority: MessagePriority = MessagePriority.NORMAL,
                     recipient: Optional[str] = None,
                     correlation_id: Optional[str] = None,
                     reply_to: Optional[str] = None,
                     ttl: Optional[int] = None,
                     **metadata) -> str:
        """发布消息"""
        try:
            # 创建消息
            message = Message(
                id=str(uuid.uuid4()),
                type=message_type,
                topic=topic,
                payload=payload,
                sender=sender,
                recipient=recipient,
                priority=priority,
                correlation_id=correlation_id,
                reply_to=reply_to,
                ttl=ttl,
                metadata=metadata
            )
            
            # 应用中间件
            message = self._apply_middleware(message)
            
            # 路由消息
            message = self._route_message(message)
            
            # 发布消息
            success = await self.backend.publish(message)
            
            if success:
                logger.debug(f"消息发布成功: {message.id} -> {message.topic}")
                return message.id
            else:
                logger.error(f"消息发布失败: {message.id}")
                return ""
                
        except Exception as e:
            logger.error(f"发布消息失败: {e}")
            return ""
    
    async def subscribe(self, topic: str, handler: Callable[[Message], Any], 
                       subscriber_id: str, filter_func: Optional[Callable[[Message], bool]] = None) -> bool:
        """订阅主题"""
        return await self.backend.subscribe(topic, handler, subscriber_id, filter_func)
    
    async def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """取消订阅"""
        return await self.backend.unsubscribe(topic, subscriber_id)
    
    async def send_command(self, target: str, command: str, params: Dict[str, Any], 
                          sender: str, timeout: int = 30) -> Optional[Any]:
        """发送命令并等待响应"""
        try:
            correlation_id = str(uuid.uuid4())
            reply_topic = f"reply.{sender}.{correlation_id}"
            
            # 创建响应等待
            response_future = asyncio.Future()
            
            def response_handler(message: Message):
                if message.correlation_id == correlation_id:
                    response_future.set_result(message.payload)
            
            # 订阅响应主题
            await self.subscribe(reply_topic, response_handler, sender)
            
            try:
                # 发送命令
                await self.publish(
                    topic=f"command.{target}",
                    payload={'command': command, 'params': params},
                    sender=sender,
                    message_type=MessageType.COMMAND,
                    priority=MessagePriority.HIGH,
                    recipient=target,
                    correlation_id=correlation_id,
                    reply_to=reply_topic,
                    ttl=timeout
                )
                
                # 等待响应
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
                
            finally:
                # 取消订阅
                await self.unsubscribe(reply_topic, sender)
                
        except asyncio.TimeoutError:
            logger.warning(f"命令超时: {command} -> {target}")
            return None
        except Exception as e:
            logger.error(f"发送命令失败: {e}")
            return None
    
    async def send_query(self, target: str, query: str, params: Dict[str, Any], 
                        sender: str, timeout: int = 10) -> Optional[Any]:
        """发送查询并等待响应"""
        return await self.send_command(target, query, params, sender, timeout)
    
    async def send_notification(self, topic: str, message: str, data: Dict[str, Any], 
                               sender: str, priority: MessagePriority = MessagePriority.NORMAL):
        """发送通知"""
        await self.publish(
            topic=f"notification.{topic}",
            payload={'message': message, 'data': data},
            sender=sender,
            message_type=MessageType.NOTIFICATION,
            priority=priority
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.backend.get_stats()
        stats['backend_type'] = 'redis' if self.use_redis else 'memory'
        stats['middleware_count'] = len(self.middleware)
        stats['route_count'] = len(self.message_routes)
        return stats


# 全局消息总线实例
message_bus = MessageBus()
