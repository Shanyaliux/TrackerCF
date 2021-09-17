from tracker.Mosse import Mosse

if __name__ == '__main__':
    img_path = 'datasets/surfer'
    tracker = Mosse(img_path)
    tracker.track()
