package au.edu.rmit;

public class HyperPoint {
    double[] coords;
    int K = 0;

    public int getK() {
        return K;
    }

    public HyperPoint(double[] crds) {
        if (crds == null)
            throw new NullPointerException("");
        K = crds.length;
        coords = new double[K];
        for (int i = 0; i < K; i++)
            coords[i] = crds[i];
    }

    public double[] getcoords(){
        return this.coords;
    }

    public HyperPoint(HyperPoint p) {
        this(p.coords);
    }

    public boolean equals(HyperPoint p) {
        if (K != p.K)
            throw new IllegalArgumentException("");
        for (int i = 0; i < K; i++)
            if (p.coords[i] != coords[i])
                return false;
        return true;
    }

    // Euclidean Distance
    public double distanceTo(HyperPoint p) {
        return Math.sqrt(squareDistanceTo(p));
    }

    public double squareDistanceTo(HyperPoint p) {
        if (K != p.K)
            throw new IllegalArgumentException("");
        double res = 0;
        for (int i = 0; i < K; i++)
            res += (coords[i] - p.coords[i]) * (coords[i] - p.coords[i]);
        return res;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < K; i++)
            sb.append(coords[i] + ",");
        return sb.toString();
    }

    public void setCoords(double[] coords) {
        this.coords = coords;
    }
}