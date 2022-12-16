Assignment (BE)

    matrix = [[3,2],
              [4,3],    
              [2,3],
              [3,4]]

    n = 30

    max_capacity = 0

    for i in range(len(matrix)):
      max_capacity+= (matrix[i][0]*matrix[i][1])

    if(n>max_capacity):
      print("Maximum", max_capacity, "seats can be booked.")
      print("Displaying result for", max_capacity, "seats")




    arr = [0 for i in range(len(matrix))]



    for i in range(len(matrix)):
      arr[i] = [[0 for k in range(matrix[i][0])] for l in range(matrix[i][1])]



    x = 1

    ####################### Aisle Seating #######################
    pos = 0
    flag = -1 

    while n>0:
      if flag == 0:
        pos = 0
        flag = -1
        break

      for i in range(len(matrix)):
        flag = 0
        if pos < len(arr[i]):
          if i == 0:
            arr[i][pos][-1] = x
            x=x+1
            n=n-1
            flag = 1
          elif i != len(matrix) - 1:
            arr[i][pos][0] = x
            x+=1
            n-=1
            flag = 1
            arr[i][pos][-1] = x
            x+=1
            n-=1
            flag = 1
          elif i == len(matrix) - 1:
            arr[i][pos][0] = x
            x+=1
            n-=1
            flag = 1

      pos+=1



    ######################## Window Seating #######################

    while n>0:
      if(flag == 0):
        pos = 0
        flag = -1
        break


      flag = 0
      if ((pos<len(arr[0]) and arr[0][pos][0] ==0)):
        arr[0][pos][0] = x
        x+=1
        n-=1
        flag = 1

      if ((pos<len(arr[-1]) and arr[-1][pos][-1] ==0)):
        arr[-1][pos][-1] = x
        x+=1
        n-=1
        flag = 1

      pos+=1



    ####################### Center Seating #######################
    while(n>0):
      if(flag == 0):
        pos=0
        flag = -1
        break

      for i in range(len(matrix)):
        flag = 0

        if(pos<len(arr[i])):
          for j in range(1,len(arr[i][pos])-1):
            if(arr[i][pos][j] == 0):
              if(n>0):
                arr[i][pos][j] = x
                x+=1
                n-=1
                flag = 1
              else:
                flag = -1


      pos+=1






    ########################### Display Output #######################
    xyz = 0
    pos=0
    flag = -1

    while (flag<=len(matrix)-1):
      flag = 0
      for i in range(len(matrix)):
        if(pos<len(arr[i])):
          if(i==0):
            print("|",end="\t")
          print(arr[i][pos],end="\t | \t")
        else:
          if(i==0):
            print("|",end="\t")
          print("\t | \t",end="\t")
          flag+=1
      pos+=1
      print("\n")



