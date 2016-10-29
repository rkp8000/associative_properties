from sqlalchemy import Column
from sqlalchemy import Integer, Float, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ErrorBound(Base):

    __tablename__ = 'error_bound'

    id = Column(Integer, primary_key=True)

    log_10_ms = Column(ARRAY(Float))
    n_mc = Column(Integer)
    n = Column(Integer)
    l = Column(Integer)
    q = Column(Float)
    r = Column(Float)

    log_errors = Column(ARRAY(Float))
    group_name = Column(String)


class ItemCapacity(Base):

    __tablename__ = 'item_capacity'

    id = Column(Integer, primary_key=True)

    max_log_10_error = Column(Float)
    n_mc = Column(Integer)
    n = Column(Integer)
    l = Column(Integer)
    q = Column(Float)
    r = Column(Float)
    log_item_capacity = Column(Float)
    group_name = Column(String)
