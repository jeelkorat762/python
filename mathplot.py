import matplotlib.pyplot as plt
import numpy as np

x=np.array([2020,2021,2022,2023,2024,2025])
y = np.array([7,8,9,10,14,20])
y_fah = (y * 9/5) + 32
k = x + 273.15


plt.title('Temperature in helsinki in last 6 years')



plt.subplot(1, 3, 1)
plt.plot(x,y, 'bo--', label = 'celsius')
plt.xlabel('Years')
plt.ylabel('Temperature (°C)')
plt.legend()


plt.subplot(1, 3, 2)
plt.plot(x,y_fah, 'r*-',label = 'fahrenheit')
plt.xlabel('Years')
plt.ylabel('Temperature (°F)')

plt.legend()

plt.subplots_adjust(wspace=0.4)

plt.subplot(1, 3, 3)
plt.plot(x,k, 'bo--', label = 'kelwin')
plt.xlabel('Years')
plt.ylabel('Temperature (°K)')
plt.legend()
plt.savefig('pic.jpg')
plt.show()