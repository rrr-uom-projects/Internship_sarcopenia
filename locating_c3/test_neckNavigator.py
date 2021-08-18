#created by hermione on 16/08/2021
#to test the model at various stages


from utils import PrintSlice, projections, setup_model
from neckNavigatorTester import neckNavigatorTest1
from neckNavigator import neckNavigator

def main():
    tester = neckNavigatorTest1(model, test_dataloader, device)
    #test_results = tester
    C3s, segments, GTs = tester
    
    print("gt info: ", len(GTs))
    print(GTs.shape,)
    print("segs info: ", segments.shape)
    #print(segments[0].shape, len(segments))
    #print(C3s[0].shape)
    #c3 = C3s[0][0]
    #segment = segments[0][0]

    difference = euclid_dis(GTs, segments)
    print(difference)
    PrintSlice(C3s[0], segments[0], show=True)
    for j in range(0,4):
        projections(C3s[j],segments[j], order = [1,2,0])

    return

if __name__ == '__main__':
    main()