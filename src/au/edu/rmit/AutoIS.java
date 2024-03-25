package au.edu.rmit;

public class AutoIS {
    IndexNode root;
    int K = 2;
    double RANGE = 1.0;
    // HyperPoint min, max are determined the range of KDTree Space
    HyperPoint min, max;
    int capacity = 500;

    public IndexNode getRoot() {
        return root;
    }
    public void setCapacity(int capacity) {
        this.capacity = capacity;
    }

    public AutoIS(int K) {
        this.K = K;
        root = null;
        double[] vals = new double[K];
        min = new HyperPoint(vals);
        for (int i = 0; i < K; i++)
            vals[i] = RANGE;
        max = new HyperPoint(vals);
    }

    public AutoIS(int K, HyperPoint min, HyperPoint max) {
        this.K = K;
        this.min = min;
        this.max = max;
        root = null;
    }

    private static int partition(HyperPoint[] points, int k, int beg, int end) {
        HyperPoint pivot = points[beg];
        int i = beg, j = end + 1;
        while (true) {
            while (++i <= end && points[i].coords[k] < pivot.coords[k])
                ;
            while (--j > beg && points[j].coords[k] >= pivot.coords[k])
                ;
            if (i < j) {
                HyperPoint temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            } else
                break;
        }
        points[beg] = points[j];
        points[j] = pivot;
        return j;
    }
    public void ConstructionAutoIS(HyperPoint[] points) {
        int num = points.length;
        HyperPoint hmin = new HyperPoint(min);
        HyperPoint hmax = new HyperPoint(max);
        root = insertByMedianFinding(root, points, hmin, hmax, 0, 0, num - 1);
    }

    public static int findMedian(HyperPoint[] points, int k, int beg, int end) {
        if (beg > end)
            return -1;
        else if (beg == end)
            return beg;
        int mid = (beg + end) / 2;
        int i = beg, j = end;
        while (true) {
            int t = partition(points, k, i, j);
            if (t == mid)
                return t;
            else if (t > mid)
                j = t - 1;
            else
                i = t + 1;
        }
    }

    public static double quickSort(HyperPoint[] arr, int begin, int end,int k) {
        if (begin < end) {
            int partitionIndex = partition_sort(arr, begin, end,k);

            quickSort(arr, begin, partitionIndex-1,k);
            quickSort(arr, partitionIndex+1, end,k);
        }
        return arr[arr.length/2].getcoords()[k];
    }

    private static int partition_sort(HyperPoint[] arr, int begin, int end,int k) {
        HyperPoint pivot = arr[end];
        int i = (begin-1);

        for (int j = begin; j < end; j++) {
            if (arr[j].getcoords()[k] <= pivot.getcoords()[k]) {
                i++;

                HyperPoint swapTemp = arr[i];
                arr[i] = arr[j];
                arr[j] = swapTemp;
            }
        }

        HyperPoint swapTemp = arr[i+1];
        arr[i+1] = arr[end];
        arr[end] = swapTemp;

        return i+1;
    }

    public static int divide(HyperPoint[] arr, double target, int l, int r,int k) {
        int left = l;
        int cur = l;
        int right = r;
        int anc = l;
        while (cur <= right) {
            if (arr[cur].getcoords()[k] < target) { // cur????????????????????????????left????
                if (cur != left) { //????????????7??????????????left?cur?????
                    swap(arr, left, cur);
                }
                cur++;
                left++;
                anc++;
            } else if (arr[cur].getcoords()[k] == target) { // ??????????????
                cur++;
                anc++;
            }else { //cur?????????????????????????????????????
                swap(arr, cur, right);
                right--;
            }
        }//1 2 5 6 7    5.5
        //if(anc == l+1) return anc;
        //if(anc == r+1) return anc-2;
        return anc - 1;
    }

    public static void swap(HyperPoint[] points, int left, int right){
        HyperPoint temp = points[left];
        points[left]= points[right];
        points[right]= temp;
    }

    private IndexNode insertByMedianFinding(IndexNode r, HyperPoint[] points, HyperPoint hmin, HyperPoint hmax, int depth, int i, int j) {
        if(j-i+1 <= capacity){
            IndexNode node = null;
            if(i > j) node = new IndexNode(new HyperSpace(hmin, hmax), points[j]);
            else node = new IndexNode(new HyperSpace(hmin, hmax), points[i]);
            node.Node_count = j-i+1;
            node.isleaf = true;
            HyperPoint[] tmp_hp = new HyperPoint[j-i+1];
            for(int anchor=0;anchor<j-i+1;anchor++){
                tmp_hp[anchor] = points[i+anchor];
            }
            node.hp = tmp_hp;
            return node;
        }
        int k = depth % K;

        int t = findMedian(points, k, i, j);
        HyperPoint p = points[t];

        if (r == null) r = new IndexNode(new HyperSpace(hmin, hmax),p);
        double pivot = p.coords[k];
        HyperPoint hmid1 = new HyperPoint(hmax);
        hmid1.coords[k] = p.coords[k];
        r.left = insertByMedianFinding(r.left, points, hmin, hmid1, depth + 1, i, t);

        HyperPoint hmid2 = new HyperPoint(hmin);
        hmid2.coords[k] = pivot;
        r.right = insertByMedianFinding(r.right, points, hmid2, hmax, depth + 1, t+1 , j);
        r.Node_count = j-i+1;
        r.pivot = pivot;
        return r;
    }

    public BoundedPQueue kNN(HyperPoint pointKD, int K){
        BoundedPQueue k_nearest_points = new BoundedPQueue(K);
        get_all_nearest_points(pointKD,k_nearest_points,root,0);

        return k_nearest_points;
    }

    private void get_all_nearest_points(HyperPoint pointKD, BoundedPQueue knp, IndexNode curr,int depth){
        if(curr == null){
            return;
        }
        if(curr.isleaf){
            for(int i=0;i<curr.hp.length;i++){
                double distance = pointKD.squareDistanceTo(curr.hp[i]);
                knp.enqueue(distance, curr.hp[i]);
            }
            return;
        }
        double value = Double.MAX_VALUE;
        if(!knp.isEmpty()){
            value = knp.worst();
        }
        int keyIndex = depth % K;
        //boolean to_left = false;// do I need to go further side, consider sheng wang kNN
        boolean near_is_left = false;
        boolean near_is_right = false;
        if(pointKD.coords[keyIndex] < curr.pivot){
            get_all_nearest_points(pointKD, knp, curr.left,depth+1);
            near_is_left = true;
        }else{
            //to_left = true;
            get_all_nearest_points(pointKD, knp, curr.right,depth+1);
            near_is_right = true;
        }

        if(knp.Size()!= knp.maxSize()||(curr.p.coords[keyIndex] - pointKD.coords[keyIndex])
                *(curr.p.coords[keyIndex] - pointKD.coords[keyIndex]) < value){
            if(!near_is_left) get_all_nearest_points(pointKD, knp, curr.left,depth+1);
            else get_all_nearest_points(pointKD, knp, curr.right,depth+1);
        }
    }
}
