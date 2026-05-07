import sqlite3
from mcp.server.fastmcp import FastMCP
import os

mcp=FastMCP("calculator")


@mcp.tool()
def calculator(first_num:float,second_num:float,operation:str):
    """
    operation : add,sub,mul,div
    """
    try:
        if operation=="add":
            result=first_num+second_num
        elif operation=="sub":
            result=first_num-second_num
        elif operation=='mul':
            result=round(first_num*second_num,2)
            print("Doing multiplication ")
        elif operation=='div':
            if not second_num:
                return {
                    "error":"Division by zero is not allowed"
                }
            result=round(first_num/second_num,2)
        return result
    except Exception as e:
        return {
            "error":str(e)
        }
    



if __name__ == "__main__":
    mcp.run()