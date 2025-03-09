
# TEMPLATE
# User (user_id PK, username, email)
# Product (product_id PK, name, price, user_id FK)

from sqlalchemy import create_engine, Column, ForeignKey
#                 app object ^
from sqlalchemy.types import (Integer, Boolean, Time, Date, DECIMAL, VARCHAR)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, sessionmaker


# define base class for table class definitions
Base = declarative_base()


# TEMPLATE
#
#class User(Base):
#    __tablename__ = 'users'
#    id = Column(Integer, primary_key=True)
#    username = Column(String)
#
#class Post(Base):
#    __tablename__ = 'posts'
#    id = Column(Integer, primary_key=True)
#    title = Column(String)
#    content = Column(String)
#    user_id = Column(Integer, ForeignKey('users.id'))
#    user = relationship('User', back_populates='posts')

# ACTUAL -- create tables
class Orders(Base):
    __tablename__ = 'orders'
    # PRIMARY KEY
    order_id = Column(VARCHAR, primary_key=True)
    
    # DATA
    order_number = Column(Integer, nullable=False)
    boxes_in_order = Column(Integer, nullable=False)
    fullfilled = Column(Boolean, nullable=False)
    date_placed = Column(Date, nullable=False)
    time_placed = Column(Time, nullable=False)
    customer_id = Column(VARCHAR, ForeignKey('customer.customer_id'), unique=True, nullable=False)
    
    # RELATION
    customer_id = relationship('Customer',back_populates='Orders')

class Customer(Base):
    __tablename__ = 'customer'
    customer_id = Column(VARCHAR, primary_key=True)
    
    # DATA
    delivery_address = Column(VARCHAR, nullable=False)
    price = Column(DECIMAL, nullable=False)
    order_id = Column(VARCHAR, ForeignKey('orders.order_id'), unique=True, nullable=False)
    
    # RELATION
    order_id = relationship('Orders',back_populates='Customer')


# set up engine based on db location
# manages the connection to the db
database_url = 'mariadb+mariadbconnector://root:toor@172.17.0.1:3306/david-warehouse-db'
engine = create_engine(database_url, echo=True)

# USE THIS ENGINE FOR A .db FILE
#database_uri = 'sqlite:///comfort_airlines.db'
#engine = create_engine(database_uri)


# create tables based on definitions above, define which engine to use
Base.metadata.create_all(bind=engine)


# declare a new session to interact with db, define which engine to use
Session = sessionmaker(bind=engine)
# create session
session = Session()


# commit changes
session.commit()


# close session
session.close()

