

"""
From this file the User Output tab is reading the data
the df is all the final results of the scheduling that we had
"""

df = [dict(Task="Job AB", Start='2009-01-01', Finish='2009-01-28'),
      dict(Task="Job B", Start='2009-01-05', Finish='2009-02-15'),
      dict(Task="Job C", Start='2009-01-20', Finish='2009-02-28')]

"""Simple function to display a jobshop solution using plotly."""
#ortools.sat.python.visualization.DisplayJobshop(starts, durations, machines, name)