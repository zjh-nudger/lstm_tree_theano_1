# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:13:18 2016

@author: tanfan.zjh
"""
import sys
def c():
    print 'hhh:'+sys._getframe().f_code.co_name  

c()