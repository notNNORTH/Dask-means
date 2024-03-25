package au.edu.rmit;

public class IndexNode {
    // HyperSpace hs is used to accelerate range search
    HyperSpace hs;

    HyperPoint p;

    boolean hasModel = false;

    IndexNode left, right;

    boolean isleaf = false;

    int ID;
    HyperPoint[] hp;
    int Node_count;

    double pivot;

    public boolean isLeaf(){
        if(left == null || right == null) return true;
        else return false;
    }

    public IndexNode(HyperSpace hs, HyperPoint p) {
        this.hs = hs;
        this.p = p;
    }

    public IndexNode(HyperSpace hs) {
        this.hs = hs;
    }

    public IndexNode() {
        this.hs = null;
        this.p = null;
    }

    public void setP(HyperPoint p) {
        this.p = p;
    }

    public HyperPoint getP() {
        return p;
    }

    public HyperSpace getHs() {
        return hs;
    }

    public boolean isHasModel() {
        return hasModel;
    }

    public void setID(int ID) {
        this.ID = ID;
    }
}
