#Make Sure to run social_distance_detector.py file before running it
import matplotlib.pyplot as plt
import numpy as np
import requests

#While Loading, ViolationCount.npy must be in the same folder
violationCount = np.load('./ViolationCount.npy')
print(violationCount.shape)

url = "https://www.fast2sms.com/dev/bulk"
payload = "sender_id=FSTSMS&message=Violation of Social Distancing Has Been Found at the GIVEN ADDRESS&language=english&route=p&numbers=9592830271,8979515765"
headers = {
'authorization': "k2HpCg55XZKFmg3qVR977QeYMbUjBLglOuf2CFWRT3BYBPjGzqrI7l1JL4aZ",
'Content-Type': "application/x-www-form-urlencoded",
'Cache-Control': "no-cache",
}
#Don't send again again, as i may go out of free credits
ViolationPerFrame = violationCount.mean()

if ViolationPerFrame > 10:
	response = requests.request("POST", url, data=payload, headers=headers)
	response = (response.text).encode('ascii', 'ignore')
	if response.find('true') != -1:
		print("Alert Successfully Sent to the Local Authority")


fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

heatmap, xedges, yedges = np.histogram2d(np.arange(len(violationCount)) ,violationCount, bins=(64,64))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
ax.imshow(heatmap, extent = extent)

ax2.plot(np.arange(len(violationCount)), violationCount)

plt.tight_layout()
plt.show()