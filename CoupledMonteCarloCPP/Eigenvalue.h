// Class for the eigenvalue
class Eigenvalue
{
  public:
    Eigenvalue(double ev); // constructor
    ~Eigenvalue(); // desctuctor

  public: // Methods
    double *get();
    void    set(double keff);

  private: // Variables
    double _keff;
};
