def partition(array, low, high):

    pivot = array[high]

    i = low - 1
 
    for j in range(low, high):

        if array[j] <= pivot:

            i = i + 1

            temp = array[j]

            array[j] = array[i]
            array[i] = temp

    temp = array[i+1]
    array[i+1] = array[high]
    array[high] = temp
 
        
    return i + 1

def quickSort(array, low, high):
    if low < high:

        pi = partition(array, low, high)

        quickSort(array, low, pi - 1)
 
        quickSort(array, pi + 1, high)


data = [87, 56, 21, 33, 92, 44, 11, 98, 67, 74, 34, 29, 13, 5, 49, 50, 83, 96, 64, 70, 18, 37, 43, 85, 2, 16, 26, 72, 55, 95, 9, 80, 65, 60, 31, 88, 7, 79, 61, 48, 22, 39, 45, 68, 93, 3, 32, 77, 84, 66, 6, 54, 12, 38, 69, 52, 86, 40, 4, 91, 20, 53, 19, 25, 57, 78, 15, 42, 97, 81, 75, 10, 73, 51, 35, 63, 24, 89, 28, 47, 8, 62, 41, 17, 23, 71, 76, 36, 46, 58, 30, 27, 90, 82, 14, 99, 59, 1, 100, 94]

print("Unsorted Array")
print(data)
 
size = len(data)

quickSort(data, 0, size - 1)
 
print('Sorted Array in Ascending Order:')
print(data)